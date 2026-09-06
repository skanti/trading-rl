"""Compatibility launcher for :mod:`trading_rl.overnight.reconcile_live_sessions`."""

import sys
from trading_rl.overnight import reconcile_live_sessions as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
