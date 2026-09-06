"""Compatibility launcher for :mod:`trading_rl.cli.download_bars`."""

import sys
from trading_rl.cli import download_bars as _implementation

if __name__ == "__main__":
    _implementation.cli()
else:
    sys.modules[__name__] = _implementation
