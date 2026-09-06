"""Compatibility launcher for :mod:`trading_rl.cli.download_nbbo`."""

import sys
from trading_rl.cli import download_nbbo as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
