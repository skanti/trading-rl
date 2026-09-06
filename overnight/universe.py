"""Compatibility launcher for :mod:`trading_rl.overnight.universe`."""

import sys
from trading_rl.overnight import universe as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
