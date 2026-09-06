"""Compatibility alias for :mod:`trading_rl.overnight.live_config`."""

import sys
from trading_rl.overnight import live_config as _implementation

sys.modules[__name__] = _implementation
