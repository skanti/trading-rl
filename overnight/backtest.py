"""Compatibility launcher for :mod:`trading_rl.overnight.backtest`."""

import sys
from trading_rl.overnight import backtest as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
