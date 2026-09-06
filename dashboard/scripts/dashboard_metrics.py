"""Compatibility import for :mod:`trading_rl.dashboard.metrics`."""

import sys

from trading_rl.dashboard import metrics as _implementation

sys.modules[__name__] = _implementation
