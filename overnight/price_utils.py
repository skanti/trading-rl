"""Compatibility alias for :mod:`trading_rl.overnight.price_utils`."""

import sys
from trading_rl.overnight import price_utils as _implementation

sys.modules[__name__] = _implementation
