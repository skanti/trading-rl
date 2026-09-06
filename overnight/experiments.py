"""Compatibility launcher for :mod:`trading_rl.overnight.experiments`."""

import sys
from trading_rl.overnight import experiments as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
