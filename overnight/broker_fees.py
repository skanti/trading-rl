"""Compatibility alias for :mod:`trading_rl.overnight.broker_fees`."""

import sys
from trading_rl.overnight import broker_fees as _implementation

sys.modules[__name__] = _implementation
