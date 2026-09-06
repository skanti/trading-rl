"""Compatibility import for :mod:`trading_rl.dashboard.config`."""

import sys

from trading_rl.dashboard import config as _implementation

sys.modules[__name__] = _implementation
