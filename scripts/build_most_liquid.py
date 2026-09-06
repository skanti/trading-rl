"""Compatibility launcher for :mod:`trading_rl.cli.build_most_liquid`."""

import sys
from trading_rl.cli import build_most_liquid as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
