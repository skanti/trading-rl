"""Exercise the real shell orchestration without downloading market data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class DownloadPipelineTest(unittest.TestCase):
    def test_nbbo_replays_strategy_shortlist_and_keeps_the_csv_override(self):
        script = Path(__file__).resolve().parents[1] / "download_latest_bars_and_auctions.sh"
        cases = (
            (False, False, "", False), (True, False, "entry", False), (True, False, "exit", False),
            (False, True, "", False), (True, False, "", False),
            (True, False, "entry_date", False), (False, False, "entry", False),
            (False, False, "", True), (True, False, "entry", True),
        )
        for use_csv, rank_fails, trade_date, auctions_fail in cases:
            with self.subTest(use_csv=use_csv, rank_fails=rank_fails, trade_date=trade_date, auctions_fail=auctions_fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner = root / "fake-python"
                runner.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                    import json, os, sys
                    from pathlib import Path
                    args = sys.argv[1:]
                    if args == ['-']:
                        print('2026-09-02')
                        raise SystemExit(0)
                    if args[0] == '-m':
                        args = args[1:]
                    name = Path(args[0]).name
                    record = {'name': name, 'args': args[1:]}
                    if name == 'download_bars.py':
                        Path(args[args.index('--out_dir') + 1]).mkdir(parents=True, exist_ok=True)
                    if name == 'build_most_liquid.py':
                        Path(args[args.index('--output') + 1]).write_text('AAPL\\nMSFT\\n')
                    if name == 'trading_rl.cli.rank':
                        assert '--auctions-path' not in args
                        assert '--minute-bars-dir' not in args
                        assert not Path(os.environ['MINUTE_BARS_DIR']).exists()
                        assert Path(args[args.index('--symbols-file') + 1]).read_text() == 'AAPL\\nMSFT\\n'
                        Path(args[args.index('--output') + 1]).write_text('MSFT\\nNVDA\\n')
                    if name == 'download_nbbo.py' and '--symbols-file' in args:
                        record['symbols'] = Path(args[args.index('--symbols-file') + 1]).read_text().splitlines()
                    with open(os.environ['TEST_COMMAND_LOG'], 'a') as log:
                        log.write(json.dumps(record) + '\\n')
                    if name == 'download_auctions.py':
                        if os.environ.get('TEST_AUCTIONS_FAIL') == '1':
                            raise SystemExit(1)
                        Path(args[args.index('--output') + 1]).write_text('refreshed')
                    if name == 'trading_rl.cli.rank' and os.environ.get('TEST_RANK_FAILS') == '1':
                        raise SystemExit(1)
                    '''))
                runner.chmod(0o755)
                trades = root / "trades.csv"
                trades.write_text("entry_date,sample_id\n2026-09-02,AAPL\n")
                (root / "auctions.npz").write_text("stale")
                env = {
                    **os.environ,
                    "PYTHON_BIN": str(runner), "ENV_FILE": str(root / "absent.env"),
                    "UPDATES_DIR": str(root), "LOCK_DIR": str(root / "lock"),
                    "MASTER_PATH": str(root / "master.txt"),
                    "LIQUIDITY_CANDIDATES_PATH": str(root / "symbols.txt"),
                    "DAILY_BARS_DIR": str(root / "daily"), "MINUTE_BARS_DIR": str(root / "minute"),
                    "AUCTIONS_PATH": str(root / "auctions.npz"), "NBBO_PATH": str(root / "nbbo.npz"),
                    "NBBO_TARGETS_PATH": str(trades) if use_csv else "",
                    "NBBO_TRADE_DATE": trade_date,
                    "ALPACA_KEY": "test", "ALPACA_SECRET": "test",
                    "ALPACA_DATA_KEY": "test", "ALPACA_DATA_SECRET": "test",
                    "TEST_COMMAND_LOG": str(root / "calls.jsonl"),
                    "TEST_RANK_FAILS": "1" if rank_fails else "0",
                    "TEST_AUCTIONS_FAIL": "1" if auctions_fail else "0",
                }
                for key in ("NBBO_RANK_SINCE", "NBBO_RANK_TOP", "NBBO_SYMBOLS_PATH"):
                    env.pop(key, None)
                result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
                self.assertFalse((root / "lock").exists())
                if (use_csv and trade_date not in ("entry", "exit")) or (not use_csv and trade_date):
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("NBBO_TRADE_DATE", result.stderr)
                    self.assertFalse((root / "calls.jsonl").exists())
                    continue
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                if auctions_fail:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(calls[-1]["name"], "download_auctions.py")
                    self.assertEqual("trading_rl.cli.rank" in [call["name"] for call in calls], not use_csv)
                    self.assertNotIn("download_nbbo.py", [call["name"] for call in calls])
                    self.assertEqual((root / "auctions.npz").read_text(), "stale")
                    continue
                if rank_fails:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(calls[-1]["name"], "trading_rl.cli.rank")
                    self.assertEqual(calls[-2]["name"], "build_most_liquid.py")
                    self.assertNotIn("download_auctions.py", [call["name"] for call in calls])
                    self.assertFalse((root / "minute").exists())
                    self.assertNotIn("download_nbbo.py", [call["name"] for call in calls])
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(calls[-1]["name"], "download_nbbo.py")
                auction_call = calls[-2]
                self.assertEqual(auction_call["name"], "download_auctions.py")
                self.assertEqual(auction_call["args"][auction_call["args"].index("--end") + 1], "2026-09-02")
                args = calls[-1]["args"]
                self.assertIn("--targets-from-trades" if use_csv else "--symbols-file", args)
                self.assertNotIn("--update", args)
                if not use_csv:
                    self.assertNotIn("--trade-date", args)
                    self.assertEqual(calls[-1]["symbols"], ["MSFT", "NVDA"])
                    self.assertEqual([call["name"] for call in calls], [
                        "universe.py", "download_bars.py", "build_most_liquid.py",
                        "trading_rl.cli.rank", "download_bars.py",
                        "download_auctions.py", "download_nbbo.py",
                    ])
                    rank_args = calls[3]["args"]
                    for flag, value in {
                        "--since": "2023-01-01", "--top": "12",
                        "--daily-bars-dir": str(root / "daily"),
                        "--symbols-file": str(root / "symbols.txt"),
                        "--output": str(root / "strategy_symbols_2023-01-01.txt"),
                    }.items():
                        self.assertEqual(rank_args[rank_args.index(flag) + 1], value)
                else:
                    self.assertEqual(args[args.index("--trade-date") + 1], trade_date)
                    self.assertNotIn("trading_rl.cli.rank", [call["name"] for call in calls])


if __name__ == "__main__":
    unittest.main()
