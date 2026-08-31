"""Overnight dollar-liquidity strategy: backtest, live execution, and universe.

``backtest`` simulates the strategy point-in-time, ``live`` runs it against an
Alpaca account, and ``universe`` maintains the tradable symbol master. The two
share their ranking definition so a simulated basket and a live one agree.
"""
