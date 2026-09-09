from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
import tempfile
import json
import numpy as np
import unittest
from unittest import mock

from trading_rl.cli import download_nbbo


def quote_row(symbol: str, day: str, ask: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": day,
        "target_timestamp": f"{day}T19:45:00Z",
        "timestamp": f"{day}T19:44:59Z",
        "bid_price": ask - 0.02,
        "ask_price": ask,
        "bid_size": 100.0,
        "ask_size": 200.0,
        "bid_exchange": "Q",
        "ask_exchange": "Q",
    }


class DownloadNbboTest(unittest.TestCase):
    def test_raw_loader_decompresses_each_column_once_and_preserves_raw_quotes(self):
        rows = [quote_row("AAPL", "2026-09-01", 100.125), quote_row("AAPL", "2026-09-02", 50.25)]
        rows[0]["timestamp"] = "2026-09-01T19:44:59.123456789Z"
        splits = [{
            "symbol": "AAPL", "ex_date": "2026-09-02", "old_rate": 1., "new_rate": 2.,
            "type": "forward_split", "id": "test-split",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nbbo.npz"
            arrays = download_nbbo._dataset_arrays(rows, splits)
            self.assertNotEqual(arrays["ask_price"][0], rows[0]["ask_price"])
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as archive:
                archive_type = type(archive)
            reads = Counter()
            original_read = archive_type.__getitem__

            def counted_read(archive, field):
                reads[field] += 1
                return original_read(archive, field)

            with mock.patch.object(archive_type, "__getitem__", counted_read):
                loaded = download_nbbo._load_raw_rows(path)
        self.assertEqual(loaded, rows)
        self.assertTrue(reads)
        self.assertEqual(max(reads.values()), 1)

    def test_trade_csv_preserves_rotating_daily_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            path.write_text(
                "entry_date,sample_id\n"
                "2026-09-01,AAPL\n"
                "2026-09-01,MSFT\n"
                "2026-09-02,MSFT\n"
                "2026-09-02,NVDA\n"
            )

            targets = download_nbbo.targets_from_trade_csv(path)

        self.assertEqual(targets[date(2026, 9, 1)], {"AAPL", "MSFT"})
        self.assertEqual(targets[date(2026, 9, 2)], {"MSFT", "NVDA"})
        self.assertEqual(set().union(*targets.values()), {"AAPL", "MSFT", "NVDA"})

    def test_downloader_queries_only_each_days_members(self):
        targets = {
            date(2026, 9, 1): {"AAPL", "MSFT"},
            date(2026, 9, 2): {"MSFT", "NVDA"},
        }

        def response(_session, _headers, _limiter, symbols, _start, target, _limit):
            stamp = target.isoformat().replace("+00:00", "Z")
            return {
                "quotes": {
                    symbol: [
                        {"t": stamp, "bp": 99.9, "ap": 100.0, "bs": 1, "as": 2}
                    ]
                    for symbol in symbols
                }
            }

        with mock.patch.object(
            download_nbbo, "_quote_request", side_effect=response
        ) as request:
            rows = download_nbbo._download_quote_rows(
                mock.Mock(),
                {},
                targets,
                time(15, 45),
                100,
                60,
                download_nbbo.RateLimiter(10_000),
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[3], ["AAPL", "MSFT"])
        self.assertEqual(request.call_args_list[1].args[3], ["MSFT", "NVDA"])

    def test_exit_targets_use_the_held_basket_on_its_exit_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            path.write_text(
                "entry_date,exit_date,sample_id\n"
                "2026-09-03,2026-09-04,AAPL\n"
                "2026-09-04,2026-09-08,MSFT\n"
            )
            targets = download_nbbo.targets_from_trade_csv(path, "exit_date")
        self.assertEqual(targets, {
            date(2026, 9, 4): {"AAPL"}, date(2026, 9, 8): {"MSFT"},
        })

    def test_missing_exit_date_does_not_fall_back_to_entry_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            path.write_text("entry_date,sample_id\n2026-09-03,AAPL\n")
            with self.assertRaisesRegex(ValueError, "exit_date"):
                download_nbbo.targets_from_trade_csv(path, "exit_date")

    def test_invalid_latest_quotes_search_earlier_pages_within_same_window(self):
        invalid = {"quotes": {"AAPL": [
            {"t": "2026-09-02T13:35:00Z", "bp": 101, "ap": 100, "bs": 1, "as": 2}
        ]}}
        valid = {"quotes": {"AAPL": [
            {"t": "2026-09-02T13:34:58Z", "bp": 99, "ap": 100, "bs": 1, "as": 2}
        ]}}
        with mock.patch.object(download_nbbo, "_quote_request", side_effect=[
            invalid, {**invalid, "next_page_token": "older"}, valid,
        ]) as request:
            rows = download_nbbo._download_quote_rows(
                mock.Mock(), {}, {date(2026, 9, 2): {"AAPL"}}, time(9, 35),
                100, 60, download_nbbo.RateLimiter(10_000),
            )
        self.assertEqual(rows[0]["timestamp"], "2026-09-02T13:34:58Z")
        self.assertEqual(request.call_count, 3)
        for call in request.call_args_list[1:]:
            self.assertEqual(call.args[4], datetime(2026, 9, 2, 13, 34, tzinfo=timezone.utc))
            self.assertEqual(call.args[5], datetime(2026, 9, 2, 13, 35, tzinfo=timezone.utc))
            self.assertEqual(call.args[6], 1000)
        self.assertEqual(request.call_args_list[2].kwargs, {"page_token": "older"})

    def test_quote_request_forwards_pagination_to_alpaca(self):
        target = datetime(2026, 9, 2, 13, 35, tzinfo=timezone.utc)
        with mock.patch.object(download_nbbo, "_request_json", return_value={}) as request:
            download_nbbo._quote_request(
                mock.Mock(), {}, mock.Mock(), ["AAPL"], target, target, 1000,
                page_token="older",
            )
        params = request.call_args.args[3]
        self.assertEqual(params["page_token"], "older")
        self.assertEqual(params["sort"], "desc")

    def test_repeated_page_token_fails_instead_of_looping_forever(self):
        with mock.patch.object(download_nbbo, "_quote_request", return_value={
            "quotes": {"AAPL": []}, "next_page_token": "same",
        }):
            with self.assertRaisesRegex(ValueError, "repeated NBBO page token"):
                download_nbbo._latest_valid_quote(
                    mock.Mock(), {}, mock.Mock(), "AAPL", date(2026, 9, 2),
                    datetime(2026, 9, 2, 13, 35, tzinfo=timezone.utc), 60,
                )

    def test_selects_latest_quote_at_or_before_target(self):
        day = date(2026, 9, 2)
        target = datetime(2026, 9, 2, 19, 45, tzinfo=timezone.utc)
        payload = {
            "quotes": {
                "AAPL": [
                    {"t": "2026-09-02T19:44:59Z", "bp": 99.9, "ap": 100.0, "bs": 1, "as": 2},
                    {"t": "2026-09-02T19:45:00Z", "bp": 100.0, "ap": 100.1, "bs": 3, "as": 4},
                    {"t": "2026-09-02T19:45:00.001Z", "bp": 100.1, "ap": 100.2, "bs": 5, "as": 6},
                ]
            }
        }

        selected = download_nbbo.select_causal_quotes(payload, day, target)

        self.assertEqual(selected["AAPL"]["ask_price"], 100.1)
        self.assertEqual(selected["AAPL"]["timestamp"], "2026-09-02T19:45:00Z")

    def test_split_adjusts_prices_and_sizes(self):
        arrays = download_nbbo._dataset_arrays(
            [quote_row("NVDA", "2024-06-07", 1_200.0)],
            [
                {
                    "type": "forward_split",
                    "symbol": "NVDA",
                    "ex_date": "2024-06-10",
                    "old_rate": "1",
                    "new_rate": "10",
                    "id": "split-1",
                }
            ],
        )

        self.assertEqual(float(arrays["ask_price"][0]), 120.0)
        self.assertEqual(float(arrays["ask_size"][0]), 2_000.0)
        self.assertEqual(float(arrays["raw_ask_price"][0]), 1_200.0)

    def test_overlap_preserves_unrequested_symbols_on_refreshed_dates(self):
        existing = [
            quote_row("AAPL", "2026-08-24", 100.0),
            quote_row("AAPL", "2026-08-25", 101.0),
            quote_row("MSFT", "2026-08-25", 500.0),
        ]
        refreshed = [quote_row("AAPL", "2026-08-25", 101.5)]

        merged = download_nbbo.merge_quote_rows(existing, refreshed, {("AAPL", "2026-08-25")})

        self.assertEqual(
            [(row["symbol"], row["date"], row["ask_price"]) for row in merged],
            [
                ("AAPL", "2026-08-24", 100.0),
                ("AAPL", "2026-08-25", 101.5),
                ("MSFT", "2026-08-25", 500.0),
            ],
        )

    def test_missing_or_overly_stale_target_warns_and_skips(self):
        target = datetime(2026, 9, 2, 19, 45, tzinfo=timezone.utc)
        stale = {
            "quotes": {
                "AAPL": [
                    {
                        "t": "2026-09-02T19:43:00Z",
                        "bp": 99.9,
                        "ap": 100.0,
                        "bs": 1,
                        "as": 2,
                    }
                ]
            }
        }
        with mock.patch.object(download_nbbo, "_quote_request", return_value=stale):
            missing = []
            with self.assertLogs(download_nbbo.LOGGER, level="WARNING") as warnings:
                rows = download_nbbo._download_quote_rows(
                    mock.Mock(),
                    {},
                    {date(2026, 9, 2): {"AAPL"}},
                    target.astimezone(download_nbbo.EASTERN).time().replace(tzinfo=None),
                    100,
                    60,
                    download_nbbo.RateLimiter(10_000),
                    missing,
                )
            self.assertEqual(rows, [])
            self.assertEqual(missing[0]["symbol"], "AAPL")
            self.assertEqual(missing[0]["date"], "2026-09-02")
            self.assertIn("Skipping NBBO 2026-09-02:AAPL", warnings.output[0])

    def test_current_snapshot_is_deferred_until_sip_delay_passes(self):
        sessions, deferred = download_nbbo.eligible_sessions(
            [date(2026, 9, 4)],
            time(15, 45),
            now=datetime(2026, 9, 4, 19, 55, tzinfo=timezone.utc),
        )

        self.assertEqual(sessions, [])
        self.assertEqual(deferred, 1)

    def test_download_writes_partial_dataset_and_missing_quote_manifest(self):
        missing = [{"date": "2026-09-02", "symbol": "AAPL", "reason": "missing"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            with mock.patch.object(download_nbbo, "_fetch", return_value=(
                [quote_row("MSFT", "2026-09-02", 100.)], 0, missing, {"2026-09-02"},
            )), mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(
                download_nbbo, "_download_splits", return_value=[]
            ):
                result = download_nbbo.download_nbbo(
                    {date(2026, 9, 2): {"AAPL", "MSFT"}}, "2026-09-02", "2026-09-02",
                    output, time(15, 45),
                )
            manifest = json.loads(output.with_suffix(".json").read_text())
            with np.load(output) as data:
                self.assertEqual(data["symbol"].tolist(), ["MSFT"])
        self.assertEqual(result, (2, 1, 0))
        self.assertEqual(manifest["missing_quote_count"], 1)
        self.assertEqual(manifest["missing_quotes"], missing)

    def test_all_missing_refresh_removes_old_quotes_but_deferred_refresh_keeps_them(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "nbbo.npz"
                download_nbbo._write_dataset(
                    output, [quote_row("AAPL", "2026-09-02", 100.)], [], ["AAPL"],
                    "2026-09-02", "2026-09-02", time(15, 45), 60, 1, 1,
                )
                missing = [] if deferred else [{"date": "2026-09-02", "symbol": "AAPL"}]
                refreshed = set() if deferred else {"2026-09-02"}
                with mock.patch.object(download_nbbo, "_fetch", return_value=(
                    [], int(deferred), missing, refreshed,
                )), mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(
                    download_nbbo, "_download_splits", return_value=[]
                ):
                    download_nbbo.update_nbbo(
                        output, "2026-09-02", {date(2026, 9, 2): {"AAPL"}}, 7, 100, 60, 180,
                    )
                self.assertEqual(len(download_nbbo._load_raw_rows(output)), int(deferred))
                manifest = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(manifest["missing_quote_count"], int(not deferred))

    def test_fetch_accounts_for_missing_targets_without_aborting(self):
        def download(_session, _headers, _targets, _target, _batch, _lookback, _limiter, missing):
            missing.append({"date": "2026-09-02", "symbol": "AAPL"})
            return []
        with mock.patch.object(download_nbbo, "_download_quote_rows", side_effect=download), mock.patch.object(
            download_nbbo, "_request_headers", return_value={}
        ), mock.patch.object(download_nbbo, "eligible_sessions", return_value=([date(2026, 9, 2)], 0)):
            rows, deferred, missing, attempted = download_nbbo._fetch(
                {date(2026, 9, 2): {"AAPL"}}, "2026-09-02", "2026-09-02", time(9, 35), 100, 60, 180,
            )
        self.assertEqual(rows, [])
        self.assertEqual(deferred, 0)
        self.assertEqual(missing[0]["symbol"], "AAPL")
        self.assertEqual(attempted, {"2026-09-02"})

    def test_symbol_file_schedule_uses_calendar_skips_early_closes_and_handles_dst(self):
        sessions = [
            {"date": "2026-03-06", "close": "16:00"},
            {"date": "2026-03-09", "close": "16:00"},
            {"date": "2026-11-27", "close": "13:00"},
        ]
        with mock.patch("trading_rl.overnight.live.load_credentials", return_value=("test", "test")), mock.patch(
            "trading_rl.overnight.live.AlpacaClient"
        ) as client:
            client.return_value.calendar.return_value = sessions
            targets = download_nbbo.targets_from_symbols({"AAPL"}, "2026-01-01", "2026-12-31", time(15, 45))
            morning = download_nbbo.targets_from_symbols({"AAPL"}, "2026-01-01", "2026-12-31", time(9, 35))
        self.assertEqual(set(targets), {date(2026, 3, 6), date(2026, 3, 9)})
        self.assertEqual(targets[date(2026, 3, 6)], {"AAPL", "SPY"})
        self.assertIn(date(2026, 11, 27), morning)
        self.assertEqual(download_nbbo.target_timestamp(date(2026, 3, 6), time(15, 45)).hour, 20)
        self.assertEqual(download_nbbo.target_timestamp(date(2026, 3, 9), time(15, 45)).hour, 19)

    def test_update_backfills_new_pairs_outside_overlap_and_preserves_other_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            old_missing = [{"date": "2026-01-02", "symbol": "MISSING", "reason": "missing"}]
            download_nbbo._write_dataset(
                output, [quote_row("AAPL", "2026-01-02", 100.), quote_row("MSFT", "2026-09-02", 200.)],
                [], ["AAPL", "MSFT", "MISSING"], "2026-01-02", "2026-09-02", time(15, 45), 60, 3, 2,
                missing_quotes=old_missing,
            )
            targets = {date(2026, 1, 2): {"AAPL", "MSFT", "MISSING"}, date(2026, 9, 2): {"AAPL"}}
            refreshed = [quote_row("MSFT", "2026-01-02", 150.), quote_row("AAPL", "2026-09-02", 110.)]
            with mock.patch.object(download_nbbo, "_fetch", return_value=(
                refreshed, 0, [], {"2026-01-02", "2026-09-02"}
            )) as fetch, mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(
                download_nbbo, "_download_splits", return_value=[]
            ):
                download_nbbo.update_nbbo(output, "2026-09-02", targets, 7, 100, 60, 180)
                self.assertEqual(fetch.call_args.args[0], {
                    date(2026, 1, 2): {"MSFT"}, date(2026, 9, 2): {"AAPL"}
                })
            rows = download_nbbo._load_raw_rows(output)
            self.assertEqual(len(rows), 4)
            self.assertIn(("MSFT", "2026-09-02", 200.), [(r["symbol"], r["date"], r["ask_price"]) for r in rows])
            manifest = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(manifest["missing_quotes"], old_missing)

    def test_missing_refresh_preserves_other_symbols_and_their_missing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            download_nbbo._write_dataset(
                output, [quote_row("AAPL", "2026-09-02", 100.), quote_row("MSFT", "2026-09-02", 200.)],
                [], ["AAPL", "MSFT", "OTHER"], "2026-09-02", "2026-09-02", time(15, 45), 60, 3, 1,
                missing_quotes=[{"symbol": "OTHER", "date": "2026-09-02"}],
            )
            with mock.patch.object(download_nbbo, "_fetch", return_value=(
                [], 0, [{"symbol": "AAPL", "date": "2026-09-02"}], {"2026-09-02"}
            )), mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(
                download_nbbo, "_download_splits", return_value=[]
            ):
                download_nbbo.update_nbbo(output, "2026-09-02", {date(2026, 9, 2): {"AAPL"}}, 7, 100, 60, 180)
            self.assertEqual([r["symbol"] for r in download_nbbo._load_raw_rows(output)], ["MSFT"])
            manifest = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual({r["symbol"] for r in manifest["missing_quotes"]}, {"AAPL", "OTHER"})

    def test_update_with_no_pending_targets_does_not_fail_or_fetch_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            download_nbbo._write_dataset(
                output, [quote_row("AAPL", "2026-01-02", 100.)], [], ["AAPL"],
                "2026-01-02", "2026-09-02", time(15, 45), 60, 1, 1,
            )
            with mock.patch.object(download_nbbo, "_fetch") as fetch, mock.patch.object(
                download_nbbo, "_request_headers", return_value={}
            ), mock.patch.object(download_nbbo, "_download_splits", return_value=[]):
                download_nbbo.update_nbbo(output, "2026-09-02", {date(2026, 1, 2): {"AAPL"}}, 7, 100, 60, 180)
            fetch.assert_not_called()
            self.assertEqual(len(download_nbbo._load_raw_rows(output)), 1)

    def test_update_api_failure_preserves_both_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            manifest = output.with_suffix(".json")
            download_nbbo._write_dataset(
                output, [quote_row("AAPL", "2026-09-02", 100.)], [], ["AAPL"],
                "2026-09-02", "2026-09-02", time(15, 45), 60, 1, 1,
            )
            before = (output.read_bytes(), manifest.read_bytes())
            with mock.patch.object(download_nbbo, "_fetch", side_effect=RuntimeError("API unavailable")):
                with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                    download_nbbo.update_nbbo(output, "2026-09-02", {date(2026, 9, 2): {"AAPL"}}, 7, 100, 60, 180)
            self.assertEqual((output.read_bytes(), manifest.read_bytes()), before)

    def test_update_extends_start_for_older_new_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nbbo.npz"
            download_nbbo._write_dataset(
                output, [quote_row("AAPL", "2026-09-02", 100.)], [], ["AAPL"],
                "2026-09-02", "2026-09-02", time(15, 45), 60, 1, 1,
            )
            with mock.patch.object(download_nbbo, "_fetch", return_value=(
                [quote_row("AAPL", "2022-01-03", 90.)], 0, [], {"2022-01-03"}
            )) as fetch, mock.patch.object(download_nbbo, "_request_headers", return_value={}), mock.patch.object(
                download_nbbo, "_download_splits", return_value=[]
            ):
                download_nbbo.update_nbbo(output, "2026-09-02", {date(2022, 1, 3): {"AAPL"}}, 7, 100, 60, 180)
            self.assertEqual(fetch.call_args.args[1], "2022-01-03")
            self.assertEqual(json.loads(output.with_suffix(".json").read_text())["start"], "2022-01-03")
            self.assertEqual(len(download_nbbo._load_raw_rows(output)), 2)


if __name__ == "__main__":
    unittest.main()
