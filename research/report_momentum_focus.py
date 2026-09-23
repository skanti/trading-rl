"""Verify the earlier optimized 8/150/40 experiment, independent of defaults."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest_plot import write_equity_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_dir
    output = root / "final-report"
    output.mkdir(parents=True, exist_ok=True)
    runs = {
        "liquidity-momentum-focus": "comparison/1-liquidity-momentum-focus",
        "liquidity-momentum-blend": "comparison/2-liquidity-momentum-blend",
        "focus + 0.25 bp/side": "focus-quarter-bp",
        "focus + 1 bp/side": "focus-one-bp",
    }
    summaries = {name: json.loads((root / folder / "summary.json").read_text()) for name, folder in runs.items()}
    selected = summaries["liquidity-momentum-focus"]
    config = selected["strategy_config"]
    if (config["allocation_window"], config["trend_window"], config["volatility_window"]) != (8, 150, 40):
        raise ValueError("this historical report requires the explicit 8/150/40 variant, not current defaults")
    audited = json.loads((root / "focus-audit/summary.json").read_text())
    pd.testing.assert_frame_equal(pd.DataFrame(selected["daily_portfolio"]), pd.DataFrame(audited["daily_portfolio"]), check_exact=True)
    screens = pd.concat([pd.read_csv(root / folder / "screen.csv") for folder in ("screen", "risk-screen", "horizon-screen")], ignore_index=True)
    row = screens.loc[(screens.allocation_window == 8) & (screens.allocation_count == 4)
                      & (screens.trend_window == 150) & (screens.volatility_window == 40)
                      & (screens.volatility_target == .35)].iloc[0]
    measured = selected["strategy_metrics"]
    np.testing.assert_allclose([row.cagr, row.sharpe, row.drawdown],
                               [measured["calendar_cagr"], measured["sharpe_zero_cash_rate"], measured["max_drawdown"]], rtol=0, atol=1e-12)
    rows, yearly = [], []
    for name, summary in summaries.items():
        values = summary["strategy_metrics"]
        record = {
            "strategy": name, "calendar_cagr": values["calendar_cagr"],
            "sharpe_zero_cash_rate": values["sharpe_zero_cash_rate"],
            "minute_mark_drawdown": summary["minute_mark_audit"]["minute_open_mark_drawdown"],
            "exit_mark_drawdown": values["max_drawdown"], "ending_equity": summary["ending_equity"],
            "trades": summary["trades"],
        }
        record["targets_met"] = bool(record["calendar_cagr"] > 1 and record["sharpe_zero_cash_rate"] >= 2.5 and record["minute_mark_drawdown"] < .2)
        rows.append(record)
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        for year, data in portfolio.groupby(pd.to_datetime(portfolio.exit_date).dt.year):
            yearly.append({"strategy": name, "exit_year": int(year),
                           "return": float(np.prod(1 + data.strategy_return) - 1),
                           "sessions": len(data)})
    if not rows[0]["targets_met"]:
        raise ValueError("frozen momentum-focus preset does not meet the requested historical targets")
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(yearly).to_csv(output / "yearly-returns.csv", index=False)
    neighborhood = screens.loc[(screens.allocation_window.isin([5, 7, 8, 10])) & (screens.allocation_count == 4)
                               & (screens.trend_window == 150) & (screens.volatility_window == 40)
                               & (screens.volatility_target == .35)]
    neighborhood.to_csv(output / "horizon-sensitivity.csv", index=False)
    chart = write_equity_plot(selected, output_dir=root / "comparison",
                              comparisons=[selected, summaries["liquidity-momentum-blend"]])
    report = {
        "status": "HISTORICAL 8/150/40 VARIANT, NOT CURRENT DEFAULT: exploratory in-sample selection using 2023-2026",
        "requested_start": "2023-01-01", "first_entry": selected["first_entry_date"],
        "last_exit": selected["last_exit_date"], "screen_trials": len(screens),
        "selection": {"config": selected["strategy_config"],
                      "rule": "lower volatility target of the two passing single-basket exit-screen configurations"},
        "results": rows, "chart": str(chart.resolve()),
        "verification": "explicit 8/150/40 variant matches audited common-engine daily portfolios; cheap screen agrees within 1e-12",
        "limitations": [
            f"No untouched holdout; {len(screens)} configurations screened in this study plus previous momentum-blend research",
            "Sharpe falls below 2.5 with an additional 0.25 bp per side",
            "Nearby 5/7/10-day horizons miss the Sharpe target; preset is sensitive to lookback selection",
            "2023 return was much weaker than 2024-2025; aggregate targets are not annual guarantees",
            "Minute-open drawdown carries stale prices and excludes intraminute extremes and executable bid-side liquidation",
            "Current security-master classification is not a complete point-in-time survivorship-free universe",
            "Broker buying-power constraints and market impact are not modeled",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Report: {(output / 'report.json').resolve()}")
    print(f"Chart: {chart.resolve()}")


if __name__ == "__main__":
    main()
