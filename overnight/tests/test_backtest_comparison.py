"""Comparisons preserve standalone results and a single aligned benchmark."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from rich.console import Console
from test_strategies import fixture

from trading_rl.overnight.backtest import (
    PreparedBacktest,
    build_parser,
    comparison_table,
    run_backtest,
    strategy_run,
)
from trading_rl.overnight.backtest_comparison import comparison_metrics, run_comparison
from trading_rl.overnight.backtest_plot import write_equity_plot
from trading_rl.overnight.backtest_report import comparison_metadata
from trading_rl.overnight.backtest_strategies import get_strategy


class BacktestComparisonTest(unittest.TestCase):
    def prepared(self, names="liquidity-momentum-blend,liquidity-fixed", extra=()):
        args = build_parser().parse_args(["--strategy", names, *extra])
        inputs = fixture()
        rows = np.arange(len(inputs["dates"]))
        inputs["daily_closes"] = np.column_stack([100 + rows, 100 + rows * 2, 100 + rows / 2])
        return PreparedBacktest(args, inputs, {})

    def summaries(self):
        prepared = self.prepared(extra=["--top", "2"])
        return [run_backtest(**strategy_run(prepared, name).inputs)[1] for name in prepared.args.strategies]

    def test_comma_and_space_syntax_preserves_order(self):
        for tokens in (["liquidity-fixed,liquidity-momentum-blend"],
                       ["liquidity-fixed,", "liquidity-momentum-blend"],
                       ["liquidity-fixed", "liquidity-momentum-blend"],
                       ["liquidity-fixed, liquidity-momentum-blend"]):
            args = build_parser().parse_args(["--strategy", *tokens])
            self.assertEqual(args.strategies, ["liquidity-fixed", "liquidity-momentum-blend"])
        for tokens in (["liquidity-fixed,liquidity-fixed"], [","], ["unknown"],
                       ["liquidity-fixed,liquidity-trend-vol", "--strategy-config", "config.json"],
                       ["liquidity-fixed,liquidity-trend-vol", "--summary-json", "summary.json"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["--strategy", *tokens])

    def test_defaults_are_per_strategy_and_arrays_are_shared(self):
        for extra, expected in (([], [(12, 10), (6, 20)]),
                                (["--top", "3", "--ema-span", "5"], [(3, 5), (3, 5)])):
            prepared = self.prepared("liquidity-fixed,liquidity-regime-vol", extra)
            for name, defaults in zip(prepared.args.strategies, expected, strict=True):
                run = strategy_run(prepared, name)
                self.assertEqual((run.inputs["top"], run.inputs["ema_span"]), defaults)
                self.assertIs(run.inputs["entry_prices"], prepared.inputs["entry_prices"])
                self.assertEqual(run.inputs["margin_interest_rate"], 0.0 if name == "liquidity-fixed" else 0.0675)
            self.assertEqual(prepared.args.strategy, "liquidity-fixed")

    def test_results_match_standalone_in_either_order(self):
        for names in ("liquidity-momentum-blend,liquidity-fixed", "liquidity-fixed,liquidity-momentum-blend"):
            prepared = self.prepared(names, ["--top", "2"])
            for name in prepared.args.strategies:
                run = strategy_run(prepared, name)
                trades, summary = run_backtest(**run.inputs)
                standalone = dict(prepared.inputs, strategy=name, strategy_config=get_strategy(name).load_config(),
                                  margin_interest_rate=0.0 if name == "liquidity-fixed" else 0.0675)
                expected_trades, expected = run_backtest(**standalone)
                pd.testing.assert_frame_equal(trades, expected_trades)
                self.assertEqual(summary["daily_portfolio"], expected["daily_portfolio"])

    def test_explicit_financing_rate_applies_to_all_strategies(self):
        prepared = self.prepared(extra=["--margin-interest-rate", "5"])
        for name in prepared.args.strategies:
            self.assertEqual(strategy_run(prepared, name).inputs["margin_interest_rate"], 0.05)

    def test_table_and_export_have_one_column_per_strategy_and_one_spy(self):
        summaries = self.summaries()
        metrics = comparison_metrics(summaries)
        self.assertEqual(list(metrics.columns), ["liquidity-momentum-blend", "liquidity-fixed", "spy-buy-and-hold"])
        self.assertEqual(metrics.loc["ending_equity", "liquidity-fixed"], summaries[1]["ending_equity"])
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None)
        table = comparison_table(summaries)
        self.assertEqual(len(table.columns), 4)  # Metric plus three results.
        console.print(table)
        text = console.export_text()
        self.assertEqual(text.count("SPY buy & hold"), 1)
        self.assertIn("liquidity-momentum-blend", text)
        metadata = comparison_metadata(summaries)
        self.assertEqual(metadata["Volatility target"], ["35.00%", "—", "—"])
        self.assertEqual(metadata["Momentum blend windows"], ["5, 10 sessions", "—", "—"])
        self.assertEqual(metadata["Short sessions"][-1], "Remains invested")
        self.assertEqual(metadata["Financing"][-1], "No borrowing")
        self.assertIn("entry/exit once", metadata["Cost"][-1])
        self.assertIn("first session only", metadata["Entry price source"][-1])
        for label in ("Period", "Position sizing", "Liquidity ranking", "Membership stability", "Stale exit marks"):
            self.assertIn(label, text)
            self.assertEqual(len(metadata[label]), 3)

    def test_misaligned_sessions_prices_or_budgets_are_rejected(self):
        summaries = self.summaries()
        for field, value in (("budget", 20000), ("entry_price_source", "minute-open")):
            changed = copy.deepcopy(summaries)
            changed[1][field] = value
            with self.assertRaisesRegex(ValueError, "identical"):
                comparison_metrics(changed)
        for field, value in (("entry_date", "1999-01-01"), ("spy_buy_and_hold_return", 0.1)):
            changed = copy.deepcopy(summaries)
            changed[1]["daily_portfolio"][0][field] = value
            with self.assertRaisesRegex(ValueError, "identical"):
                comparison_metrics(changed)

    def test_combined_plot_contains_each_strategy_and_spy_once(self):
        summaries = self.summaries()
        labels = []
        original = Axes.plot

        def recording_plot(axis, *args, **kwargs):
            labels.append(kwargs.get("label", ""))
            return original(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Axes, "plot", recording_plot):
            path = write_equity_plot(summaries[0], Path(directory), comparisons=summaries)
            self.assertTrue(path.is_file())
        self.assertEqual(len(labels), 3)
        self.assertTrue(labels[0].startswith("liquidity-momentum-blend"))
        self.assertTrue(labels[1].startswith("liquidity-fixed"))
        self.assertTrue(labels[2].startswith("SPY buy & hold"))

    def test_comparison_writes_separate_runs_and_combined_artifacts(self):
        prepared = self.prepared(extra=["--top", "2"])
        summaries = self.summaries()
        with tempfile.TemporaryDirectory() as directory:
            prepared.args.output_dir = Path(directory)
            with patch("trading_rl.overnight.backtest.execute_prepared", side_effect=summaries) as execute:
                run_comparison(prepared)
            paths = [call.args[0].args.output_dir for call in execute.call_args_list]
            self.assertEqual(len(set(paths)), 2)
            for name in ("comparison.csv", "comparison.json"):
                self.assertTrue((Path(directory) / name).is_file())
            exported = pd.read_csv(Path(directory) / "comparison.csv", index_col="metric")
            self.assertEqual(exported.loc["Momentum blend windows", "liquidity-momentum-blend"], "5, 10 sessions")
            report = json.loads((Path(directory) / "comparison.json").read_text())
            self.assertEqual(report["metadata"]["spy-buy-and-hold"]["Financing"], "No borrowing")
            self.assertIsNone(prepared.args.output_csv)


if __name__ == "__main__":
    unittest.main()
