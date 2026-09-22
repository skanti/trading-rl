"""Export one production-prepared panel for repeatable offline research sweeps."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_rl.overnight.backtest import prepare_backtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, backtest_args = parser.parse_known_args()
    prepared = prepare_backtest(backtest_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, values = {}, {}
    for key, value in prepared.inputs.items():
        if isinstance(value, pd.DatetimeIndex):
            arrays[key] = value.to_numpy(dtype="datetime64[D]")
        elif isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            values[key] = value.as_dict() if hasattr(value, "as_dict") else str(value) if isinstance(value, pd.Timestamp) else value
    np.savez_compressed(args.output_dir / "inputs.npz", **arrays)
    (args.output_dir / "inputs.json").write_text(json.dumps(values, indent=2) + "\n")
    manifest = {"backtest_args": backtest_args, "provenance": prepared.provenance,
                "sha256": {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                           for name in ("inputs.npz", "inputs.json")}}
    (args.output_dir / "inputs-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(prepared.inputs['dates'])} dates in {args.output_dir}")


if __name__ == "__main__":
    main()
