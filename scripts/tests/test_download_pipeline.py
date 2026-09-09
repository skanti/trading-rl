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
    def test_nbbo_runs_without_a_csv_and_keeps_the_csv_override(self):
        script = Path(__file__).resolve().parents[1] / "download_latest_bars_and_auctions.sh"
        for use_csv in (False, True):
            with self.subTest(use_csv=use_csv), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner = root / "fake-python"
                runner.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                    import json, os, sys
                    from pathlib import Path
                    args = sys.argv[1:]
                    if args == ['-']:
                        print('2026-09-02')
                        raise SystemExit(0)
                    name = Path(args[0]).name
                    record = {'name': name, 'args': args[1:]}
                    if name == 'download_bars.py':
                        Path(args[args.index('--out_dir') + 1]).mkdir(parents=True, exist_ok=True)
                    if name == 'build_most_liquid.py':
                        Path(args[args.index('--output') + 1]).write_text('AAPL\\nMSFT\\n')
                    if name == 'download_nbbo.py' and '--symbols-file' in args:
                        record['symbols'] = Path(args[args.index('--symbols-file') + 1]).read_text().splitlines()
                    with open(os.environ['TEST_COMMAND_LOG'], 'a') as log:
                        log.write(json.dumps(record) + '\\n')
                    '''))
                runner.chmod(0o755)
                trades = root / "trades.csv"
                trades.write_text("entry_date,sample_id\n2026-09-02,AAPL\n")
                env = {
                    **os.environ,
                    "PYTHON_BIN": str(runner), "ENV_FILE": str(root / "absent.env"),
                    "UPDATES_DIR": str(root), "LOCK_DIR": str(root / "lock"),
                    "MASTER_PATH": str(root / "master.txt"),
                    "LIQUIDITY_CANDIDATES_PATH": str(root / "symbols.txt"),
                    "DAILY_BARS_DIR": str(root / "daily"), "MINUTE_BARS_DIR": str(root / "minute"),
                    "AUCTIONS_PATH": str(root / "auctions.npz"), "NBBO_PATH": str(root / "nbbo.npz"),
                    "NBBO_TARGETS_PATH": str(trades) if use_csv else "",
                    "ALPACA_KEY": "test", "ALPACA_SECRET": "test",
                    "ALPACA_DATA_KEY": "test", "ALPACA_DATA_SECRET": "test",
                    "TEST_COMMAND_LOG": str(root / "calls.jsonl"),
                }
                subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True, text=True)
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(calls[-1]["name"], "download_nbbo.py")
                args = calls[-1]["args"]
                self.assertIn("--targets-from-trades" if use_csv else "--symbols-file", args)
                self.assertNotIn("--update", args)
                if not use_csv:
                    self.assertEqual(calls[-1]["symbols"], ["SPY", "AAPL", "MSFT"])
                self.assertFalse((root / "lock").exists())


if __name__ == "__main__":
    unittest.main()
