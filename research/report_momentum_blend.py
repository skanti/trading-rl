"""Summarize the final experimental preset without concealing selection reuse."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from research.regime_screen import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.study_dir / "final-report"
    output.mkdir(parents=True, exist_ok=True)
    runs = {
        "liquidity-trend-vol": "evaluation/liquidity-trend-vol",
        "liquidity-momentum-blend": "blend-default-inputs",
        "blend + 0.25 bp/side": "blend-quarter-bp",
        "blend + 1 bp/side": "liquidity-momentum-blend-1bp",
    }
    table, yearly, portfolios = [], [], {}
    folds = {}
    for name, directory in runs.items():
        summary = json.loads((args.study_dir / directory / "summary.json").read_text())
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        portfolios[name] = portfolio
        table.append({"strategy": name, "calendar_cagr": summary["strategy_metrics"]["calendar_cagr"],
                      "minute_mark_drawdown": summary["minute_mark_audit"]["minute_open_mark_drawdown"],
                      "exit_mark_drawdown": summary["strategy_metrics"]["max_drawdown"],
                      "sharpe": summary["strategy_metrics"]["sharpe_zero_cash_rate"],
                      "ending_equity": summary["ending_equity"], "trades": summary["trades"]})
        for year, group in portfolio.groupby(pd.to_datetime(portfolio.exit_date).dt.year):
            yearly.append({"strategy": name, "year": int(year), "return": float(np.prod(1 + group.strategy_return) - 1)})
        folds[name] = {}
        for fold, mask in {"2023-2024 entry sessions": portfolio.entry_date < "2025-01-01",
                           "2025 entry sessions": (portfolio.entry_date >= "2025-01-01") & (portfolio.entry_date < "2026-01-01"),
                           "2026 entry sessions (reused)": portfolio.entry_date >= "2026-01-01"}.items():
            data = portfolio.loc[mask]
            folds[name][fold] = metrics(data.strategy_return, data.entry_date.to_numpy(), data.exit_date.to_numpy())
    pd.DataFrame(table).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(yearly).to_csv(output / "yearly-returns.csv", index=False)
    (output / "report.json").write_text(json.dumps({
        "selection_status": "IN-SAMPLE: final preset chosen after reusing 2026; no independent future validation",
        "constraints": "2x overnight cap; historical CAGR >100% and minute-mark drawdown <=20%",
        "results": table, "folds": folds,
        "execution": "15:45 SIP ask to next primary opening auction; fractional sizing; 6.75% margin interest",
        "limitations": ["multiple-testing and parameter selection bias", "current security-master classification/survivorship limitations",
                        "minute-open marks omit intraminute extremes and carry stale prints", "actual broker limits and market impact not modeled",
                        "1 additional bp per side drops CAGR below the target"],
    }, indent=2) + "\n")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, portfolio in portfolios.items():
        if name == "blend + 0.25 bp/side":
            continue
        dates = pd.to_datetime(portfolio.exit_date)
        equity = portfolio.portfolio_end_equity.to_numpy()
        peak = np.maximum.accumulate(np.r_[10000, equity])[1:]
        axes[0].plot(dates, equity, label=name)
        axes[1].plot(dates, 100 * (equity / peak - 1), label=name)
    axes[0].set_title("Experimental momentum blend — historical, in-sample comparison")
    axes[0].set_ylabel("Equity ($10,000 start)")
    axes[1].set_ylabel("Exit-mark drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    print(pd.DataFrame(table).to_string(index=False))


if __name__ == "__main__":
    main()
