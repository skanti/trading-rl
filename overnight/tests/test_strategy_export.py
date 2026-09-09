from contextlib import redirect_stderr
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from rich.console import Console

from trading_rl.cli.download_nbbo import targets_from_trade_csv
from trading_rl.overnight import backtest, history, ranking
from trading_rl.cli import rank
from trading_rl.market_data.schema import BAR_COLUMNS, BAR_SCHEMA_VERSION
from overnight.tests.test_backtest import write_auction_npz


def ranking_inputs():
    dates = pd.date_range("2026-07-01", periods=30, freq="B")
    symbols = np.asarray(["SPY", "AAPL", "GOOG", "GOOGL", "MSFT", "NVDA"])
    volume = np.asarray(
        [
            [
                1_000,
                2_000 + i * 20,
                3_000 - i * 5,
                3_200 - i * 15,
                1_000 + i * 30,
                1_500 + i * 40,
            ]
            for i in range(len(dates))
        ],
        dtype=float,
    )
    prices = np.full_like(volume, 100.0)
    staleness = np.zeros_like(volume)
    exchange_mask = np.ones_like(volume, dtype=bool)
    exchange_mask[15:20, 3] = False
    sessions = np.ones(len(dates), dtype=bool)
    sessions[18] = False
    return dict(
        dates=dates,
        symbols=symbols,
        dollar_volume=volume,
        entry_prices=prices,
        entry_staleness=staleness,
        start_date=dates[10],
        end_date=dates[-1],
        top=2,
        ema_span=3,
        min_history_days=3,
        minimum_trading_days=5,
        max_entry_staleness_minutes=10,
        issuers={"GOOG": "Alphabet", "GOOGL": "Alphabet"},
        execution_exchange_mask=exchange_mask,
        entry_session_mask=sessions,
    )


def replay_inputs(inputs, *, price_eligibility=False):
    result = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            "entry_prices",
            "entry_staleness",
            "entry_price_source",
            "max_entry_staleness_minutes",
        }
    }
    if price_eligibility:
        result["eligibility_mask"] = (
            np.isfinite(inputs["entry_prices"])
            & (inputs["entry_prices"] > 0)
            & (inputs["entry_staleness"] <= inputs["max_entry_staleness_minutes"])
        )
    return result


class StrategyExportTest(unittest.TestCase):
    def test_replay_matches_backtest_members_and_ranks_for_both_strategies(self):
        for scheme in ("dollar_ema", "turnover_stability"):
            for source in ("minute-open", "nbbo-ask"):
                with self.subTest(scheme=scheme, source=source):
                    inputs = ranking_inputs()
                    if source == "minute-open":
                        inputs["entry_staleness"][12, 3] = 11
                    inputs.update(liquidity_scheme=scheme, entry_price_source=source)
                    selections = ranking.replay_strategy_selections(
                        **replay_inputs(inputs, price_eligibility=source != "nbbo-ask")
                    )
                    trades, _ = backtest.run_backtest(
                        **inputs,
                        morning_prices=np.full_like(inputs["entry_prices"], 101.0),
                        morning_staleness=np.zeros_like(inputs["entry_staleness"]),
                        transaction_cost_bps=0.0,
                        max_exit_staleness_minutes=1,
                    )
                    self.assertEqual(list(selections.sample_id), list(trades.sample_id))
                    self.assertEqual(list(selections["rank"]), list(trades["rank"]))
                    self.assertEqual(
                        list(selections.entry_date), list(trades.entry_date.astype(str))
                    )
                    self.assertEqual(
                        list(selections.exit_date), list(trades.exit_date.astype(str))
                    )
                    np.testing.assert_allclose(
                        selections.liquidity_score, trades.liquidity_score
                    )
                    self.assertNotIn("SPY", selections.sample_id.tolist())
                    self.assertNotIn(
                        str(inputs["dates"][18].date()), selections.entry_date.tolist()
                    )
                    for _, basket in selections.groupby("entry_date"):
                        self.assertLessEqual(
                            len(set(basket.sample_id) & {"GOOG", "GOOGL"}), 1
                        )

    def test_nbbo_export_keeps_intended_members_without_quotes(self):
        inputs = ranking_inputs()
        exported = ranking.replay_strategy_selections(**replay_inputs(inputs))
        scores = ranking.liquidity_scores(
            inputs["dollar_volume"],
            "turnover_stability",
            inputs["ema_span"],
            inputs["min_history_days"],
        )
        completed = ranking.causal_completed_trading_days(inputs["dollar_volume"])
        for day, basket in exported.groupby("entry_date"):
            index = inputs["dates"].get_loc(pd.Timestamp(day))
            selected = backtest.select_strategy_basket(
                inputs["symbols"],
                scores[index],
                completed[index],
                np.full(len(inputs["symbols"]), np.nan),
                np.full(len(inputs["symbols"]), np.inf),
                top=inputs["top"],
                minimum_trading_days=inputs["minimum_trading_days"],
                max_entry_staleness_minutes=inputs["max_entry_staleness_minutes"],
                entry_price_source="nbbo-ask",
                issuers=inputs["issuers"],
                execution_exchange_mask=inputs["execution_exchange_mask"][index + 1],
            )
            self.assertEqual(
                inputs["symbols"][selected].tolist(), basket.sample_id.tolist()
            )

    def test_later_and_same_day_liquidity_cannot_change_earlier_selection(self):
        inputs = ranking_inputs()
        original = ranking.replay_strategy_selections(**replay_inputs(inputs))
        inputs["dollar_volume"][15:, 1] *= 1_000_000
        modified = ranking.replay_strategy_selections(**replay_inputs(inputs))
        cutoff = str(inputs["dates"][15].date())
        pd.testing.assert_frame_equal(
            original[original.entry_date <= cutoff],
            modified[modified.entry_date <= cutoff],
        )

    def test_export_writes_unique_symbols_and_nbbo_compatible_entry_and_exit_targets(
        self,
    ):
        selections = ranking.replay_strategy_selections(
            **replay_inputs(ranking_inputs())
        )
        with tempfile.TemporaryDirectory() as directory:
            symbols_path, csv_path = (
                Path(directory) / "symbols.txt",
                Path(directory) / "targets.csv",
            )
            unique = rank.write_strategy_export(selections, symbols_path, csv_path)
            self.assertEqual(
                symbols_path.read_text().splitlines(), sorted(set(selections.sample_id))
            )
            self.assertEqual(unique, sorted(set(selections.sample_id)))
            for column in ("entry_date", "exit_date"):
                targets = targets_from_trade_csv(csv_path, column)
                self.assertEqual(set().union(*targets.values()), set(unique))
                self.assertTrue(all(len(basket) == 2 for basket in targets.values()))
            before = symbols_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "must differ"):
                rank.write_strategy_export(selections, symbols_path, symbols_path)
            self.assertEqual(symbols_path.read_bytes(), before)

    def test_cli_has_ranking_options_and_rejects_execution_and_removed_export_options(
        self,
    ):
        defaults = rank.build_parser().parse_args([])
        backtest_defaults = backtest.build_parser().parse_args([])
        for name in (
            "top",
            "ema_span",
            "min_history_days",
            "min_trading_days",
            "liquidity_scheme",
            "asset_filter",
            "exchange_filter",
            "unclassified_asset_policy",
            "entry_time",
            "dedupe_share_classes",
        ):
            self.assertEqual(getattr(defaults, name), getattr(backtest_defaults, name))
        self.assertEqual(defaults.output, Path("/tmp/strategy_symbols.txt"))
        for option, value in (
            ("--entry-price-source", "nbbo-ask"),
            ("--exit-price-source", "minute-open"),
            ("--nbbo-path", "missing.npz"),
            ("--budget", "10000"),
            ("--transaction-cost-bps", "0"),
        ):
            with (
                self.subTest(option=option),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                rank.build_parser().parse_args([option, value])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            backtest.build_parser().parse_args(["--export-symbols", "/tmp/removed.txt"])


class RankingCliTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = ranking_inputs()
        self.daily = self.root / "daily"
        self.minute = self.root / "minute"
        for path, timeframe in ((self.daily, "1Day"), (self.minute, "1Min")):
            path.mkdir()
            (path / "_download_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": BAR_SCHEMA_VERSION,
                        "timeframe": timeframe,
                        "adjustment": "split",
                        "columns": list(BAR_COLUMNS[timeframe]),
                    }
                )
            )
        dates = self.inputs["dates"]
        daily_seconds = dates.tz_localize("America/New_York").tz_convert("UTC").as_unit(
            "s"
        ).asi8 - int(pd.Timestamp("2010-01-01", tz="UTC").timestamp())
        master = {}
        minute_rows = []
        auction_rows = []
        for i, stamp in enumerate(dates):
            close = "13:00" if not self.inputs["entry_session_mask"][i] else "16:00"
            stamps = pd.date_range(
                f"{stamp.date()} 09:30",
                f"{stamp.date()} {close}",
                freq="min",
                tz="America/New_York",
            )
            seconds = stamps.tz_convert("UTC").as_unit("s").asi8 - int(
                pd.Timestamp("2010-01-01", tz="UTC").timestamp()
            )
            minute_rows.extend(
                [[value, 1000, 1000, 1000, 1000, 100, 10, 1000] for value in seconds]
            )
            auction_rows.append(
                dict(
                    symbol="SPY",
                    date=str(stamp.date()),
                    session="close",
                    condition="6",
                    price=1.0,
                    size=100,
                    exchange="P",
                    timestamp=stamps[-1].isoformat(),
                )
            )
            for j, symbol in enumerate(self.inputs["symbols"]):
                auction_rows.append(
                    dict(
                        symbol=symbol,
                        date=str(stamp.date()),
                        session="open",
                        condition="O",
                        price=1.0,
                        size=100,
                        exchange="Q"
                        if self.inputs["execution_exchange_mask"][i, j]
                        else "N",
                    )
                )
        for j, symbol in enumerate(self.inputs["symbols"]):
            rows = [
                [
                    seconds,
                    1000,
                    1000,
                    1000,
                    1000,
                    int(self.inputs["dollar_volume"][i, j]),
                    10,
                    1000,
                ]
                for i, seconds in enumerate(daily_seconds)
            ]
            np.save(self.daily / f"{symbol}.npy", np.asarray(rows, dtype=np.int64))
            if symbol != "SPY":
                # The candidate inventory must be available before NBBO collection,
                # but ranking must never open stock minute files or inspect prices.
                (self.minute / f"{symbol}.npy").write_bytes(b"not execution data")
            master[symbol] = {
                "name": symbol + " - Common Stock",
                "exchange": "Q",
                "etf": "N",
                "test_issue": "N",
            }
        master["GOOG"]["name"] = "Alphabet Inc. - Class C Capital Stock"
        master["GOOGL"]["name"] = "Alphabet Inc. - Class A Common Stock"
        np.save(self.minute / "SPY.npy", np.asarray(minute_rows, dtype=np.int32))
        self.auctions = self.root / "auctions.npz"
        self.auction_rows = auction_rows
        write_auction_npz(self.auctions, auction_rows)
        self.master_path = self.root / "master.json"
        self.master_path.write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "securities": master,
                }
            )
        )
        self.argv = [
            "trading-rank",
            "--since",
            str(dates[10].date()),
            "--top",
            "2",
            "--ema-span",
            "3",
            "--min-history-days",
            "3",
            "--min-trading-days",
            "5",
            "--minute-bars-dir",
            str(self.minute),
            "--daily-bars-dir",
            str(self.daily),
            "--auctions-path",
            str(self.auctions),
            "--security-master-cache",
            str(self.master_path),
            "--output",
            str(self.root / "symbols.txt"),
        ]
        self.output = io.StringIO()

    def run_cli(self, argv=None):
        with (
            mock.patch("sys.argv", argv or self.argv),
            mock.patch.object(rank, "Console", return_value=Console(file=self.output)),
            mock.patch.object(
                history.requests,
                "get",
                side_effect=AssertionError("must use cached security master"),
            ),
            mock.patch.object(
                backtest,
                "load_or_build_cache",
                side_effect=AssertionError("must not build execution cache"),
            ),
            mock.patch.object(
                backtest,
                "run_backtest",
                side_effect=AssertionError("must not simulate"),
            ),
            mock.patch.object(
                backtest,
                "load_opening_auction_prices",
                side_effect=AssertionError("must not price exits"),
            ),
            mock.patch.object(
                backtest,
                "load_scheduled_nbbo_prices",
                side_effect=AssertionError("must not load quotes"),
            ),
        ):
            rank.main()

    def test_cli_exports_intended_baskets_without_stock_minute_prices_or_nbbo(self):
        self.run_cli()
        expected = ranking.replay_strategy_selections(**replay_inputs(self.inputs))
        actual = pd.read_csv(self.root / "symbols.csv")
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(
            (self.root / "symbols.txt").read_text().splitlines(),
            sorted(set(expected.sample_id)),
        )
        self.assertIn("Shortened entries skipped", self.output.getvalue())
        self.assertNotIn(
            str(self.inputs["dates"][18].date()), actual.entry_date.tolist()
        )
        self.assertEqual(actual.exit_date.max(), str(self.inputs["dates"][-1].date()))
        for column in ("entry_date", "exit_date"):
            self.assertTrue(targets_from_trade_csv(self.root / "symbols.csv", column))

    def test_insufficient_warmup_fails_with_date_and_preserves_exports(self):
        (self.root / "symbols.txt").write_text("existing symbols\n")
        (self.root / "symbols.csv").write_text("existing schedule\n")
        argv = self.argv.copy()
        argv[argv.index("--since") + 1] = str(self.inputs["dates"][0].date())
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as failure:
            self.run_cli(argv)
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("cannot rank 2026-07-01", error.getvalue())
        self.assertIn("5 completed sessions", error.getvalue())
        self.assertEqual((self.root / "symbols.txt").read_text(), "existing symbols\n")
        self.assertEqual((self.root / "symbols.csv").read_text(), "existing schedule\n")

    def test_missing_official_session_close_is_a_hard_error(self):
        missing_date = str(self.inputs["dates"][12].date())
        write_auction_npz(
            self.auctions,
            [
                row
                for row in self.auction_rows
                if not (row["session"] == "close" and row["date"] == missing_date)
            ],
        )
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_cli()
        self.assertIn("official session close is unavailable", error.getvalue())
        self.assertIn(missing_date, error.getvalue())
        self.assertFalse((self.root / "symbols.txt").exists())

    def test_date_cutoff_and_custom_csv_are_respected(self):
        end = self.inputs["dates"][20]
        destination = self.root / "daily_targets.csv"
        self.run_cli(
            [
                *self.argv,
                "--end-date",
                str(end.date()),
                "--output-csv",
                str(destination),
            ]
        )
        actual = pd.read_csv(destination)
        self.assertEqual(actual.exit_date.max(), str(end.date()))
        self.assertLess(actual.entry_date.max(), str(end.date()))
        self.assertFalse((self.root / "symbols.csv").exists())


if __name__ == "__main__":
    unittest.main()
