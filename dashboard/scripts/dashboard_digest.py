"""Compatibility import for :mod:`trading_rl.dashboard.digest`."""

import sys

from trading_rl.dashboard import digest as _implementation

sys.modules[__name__] = _implementation
