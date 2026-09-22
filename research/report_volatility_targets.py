"""Compare 30% target runs with unchanged 35% strategy presets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest_plot import write_equity_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    directory = parser.parse_args().study_dir
    runs = (
        ("liquidity-trend-vol", 0.35, "default35/1-liquidity-trend-vol"),
        ("liquidity-trend-vol", 0.30, "trend30"),
        ("liquidity-momentum-blend", 0.35, "default35/2-liquidity-momentum-blend"),
        ("liquidity-momentum-blend", 0.30, "blend30"),
    )
    originals, rows, charts = {}, [], []
    for strategy, target, location in runs:
        path = directory / location / "summary.json"
        summary = json.loads(path.read_text())
        assert summary["strategy"] == strategy
        assert summary["strategy_config"]["volatility_target"] == target
        config = {key: value for key, value in summary["strategy_config"].items() if key != "volatility_target"}
        portfolio = pd.DataFrame(summary["daily_portfolio"])
        if target == 0.35:
            originals[strategy] = (config, summary, portfolio)
        else:
            original_config, original, original_portfolio = originals[strategy]
            assert config == original_config, "only the target may change"
            for key in ("budget", "transaction_cost_bps_per_side", "margin_interest_rate",
                        "entry_price_source", "exit_price_source", "share_mode", "top", "ema_span_sessions"):
                assert summary[key] == original[key], key
            pd.testing.assert_frame_equal(
                portfolio[["entry_date", "exit_date", "unscaled_return", "spy_buy_and_hold_return"]],
                original_portfolio[["entry_date", "exit_date", "unscaled_return", "spy_buy_and_hold_return"]],
                check_exact=True,
            )
            shared_paths = summary["input_sha256"].keys() & original["input_sha256"].keys()
            assert all(summary["input_sha256"][key] == original["input_sha256"][key] for key in shared_paths)
        audit = summary["minute_mark_audit"]
        metrics = summary["strategy_metrics"]
        rows.append({
            "strategy": strategy, "volatility_target": target,
            "calendar_cagr": metrics["calendar_cagr"],
            "annualized_return_252": metrics["annualized_return"],
            "minute_mark_drawdown": audit["minute_open_mark_drawdown"],
            "exit_mark_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe_zero_cash_rate"],
            "realized_annualized_volatility": metrics["annualized_volatility"],
            "mean_exposure": summary["average_exposure"],
            "maximum_exposure": summary["maximum_exposure"],
            "annual_borrow_drag": summary["annual_borrow_drag"],
            "ending_equity": summary["ending_equity"],
            "total_return": metrics["total_return"],
            "sessions": metrics["periods"], "trades": summary["trades"],
            "summary_path": str(path),
        })
        charts.append(dict(summary, strategy=f"{strategy} ({target:.0%} target)"))
    table = pd.DataFrame(rows)
    table.to_csv(directory / "comparison.csv", index=False)
    plot = write_equity_plot(charts[0], output_dir=directory, comparisons=charts)
    report = {
        "first_entry_date": charts[0]["first_entry_date"], "last_exit_date": charts[0]["last_exit_date"],
        "verification": "Only volatility_target changed; modeled unit returns and shared input hashes match exactly.",
        "selection_status": "Historical sensitivity comparison; momentum blend selected in-sample on reused history.",
        "results": rows, "plot_path": str(plot),
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()
    print(table.drop(columns=["summary_path"]).to_string(index=False))
    print(f"Saved comparison chart to {plot}")


if __name__ == "__main__":
    main()
