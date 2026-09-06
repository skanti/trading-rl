"""Compatibility launcher for :mod:`trading_rl.dashboard.auth`."""

import sys

from trading_rl.dashboard import auth as _implementation

if __name__ == "__main__":
    sys.exit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
