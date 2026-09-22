"""Aligned, reproducible comparisons of backtester strategy families."""

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from .backtest_report import comparison_metadata


def comparison_metrics(summaries: list[dict]) -> pd.DataFrame:
    """Reject mismatched runs before reporting a single common SPY benchmark."""
    first = summaries[0]
    reference = pd.DataFrame(first["daily_portfolio"])
    common = (
        "budget", "share_mode", "entry_price_source", "exit_price_source",
        "entry_time_eastern", "exit_time_eastern", "transaction_cost_bps_per_side",
        "first_entry_date", "last_exit_date",
    )
    columns = {}
    for summary in summaries:
        if any(summary[key] != first[key] for key in common):
            raise ValueError("strategy comparison requires identical dates, capital and execution assumptions")
        sessions = pd.DataFrame(summary["daily_portfolio"])
        if not sessions[["entry_date", "exit_date"]].equals(reference[["entry_date", "exit_date"]]):
            raise ValueError("strategy comparison requires identical reporting sessions")
        if not np.array_equal(sessions.spy_buy_and_hold_return, reference.spy_buy_and_hold_return, equal_nan=True):
            raise ValueError("strategy comparison requires identical SPY benchmark returns")
        columns[summary["strategy"]] = dict(
            summary["strategy_metrics"], ending_equity=summary["ending_equity"],
            trades=summary["trades"], average_exposure=summary["average_exposure"],
        )
    capital = first["budget"] if first["budget"] is not None else 1.0
    benchmark = first["spy_buy_and_hold_metrics"]
    years = (pd.Timestamp(first["last_exit_date"]) - pd.Timestamp(first["first_entry_date"])).days / 365.25
    columns["spy-buy-and-hold"] = dict(
        benchmark, ending_equity=capital * (1 + benchmark["total_return"]),
        trades=int(benchmark["periods"] > 0), average_exposure=1.0,
        calendar_cagr=(1 + benchmark["total_return"]) ** (1 / years) - 1,
    )
    return pd.DataFrame(columns).rename_axis("metric")


def run_comparison(prepared) -> list[dict]:
    from .backtest import comparison_table, execute_prepared, strategy_run
    from .backtest_plot import write_equity_plot

    output_dir = prepared.args.output_dir
    if output_dir is None:
        root = Path("/tmp/trading-backtests/candidate/comparisons")
        root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="comparison_", dir=root))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    console = Console()
    for index, name in enumerate(prepared.args.strategies, 1):
        console.print(f"Running {index}/{len(prepared.args.strategies)}: {name}")
        run = strategy_run(prepared, name)
        # Keep names recognizable while preventing plugin names from escaping the directory.
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        run.args.output_dir = output_dir / f"{index}-{safe_name}"
        summaries.append(execute_prepared(run, print_report=False))
    metrics = comparison_metrics(summaries)
    metadata = pd.DataFrame(comparison_metadata(summaries), index=metrics.columns).T.rename_axis("metric")
    pd.concat([metrics, metadata]).to_csv(output_dir / "comparison.csv")
    report = {
        "strategies": [summary["strategy"] for summary in summaries],
        "metrics": metrics.to_dict(),
        "metadata": metadata.to_dict(),
        "runs": summaries,
    }
    try:
        report["plot_path"] = str(write_equity_plot(summaries[0], output_dir=output_dir, comparisons=summaries).resolve())
    except Exception as error:  # noqa: BLE001 - preserve numerical results on plotting failure
        console.print(f"Could not write comparison plot: {error}")
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    console.print(comparison_table(summaries))
    console.print(f"Saved comparison and individual runs to {output_dir}")
    if "plot_path" in report:
        console.print(f"Saved comparison chart to {report['plot_path']}", markup=False, soft_wrap=True)
    return summaries
