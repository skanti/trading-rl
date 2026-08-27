/**
 * Shapes written by `scripts/dashboard_daemon.py`. Keep these in step with its
 * `build_snapshot` function -- that daemon is the only
 * writer, so it defines the contract.
 */

export type BucketKey = 'today' | 'week' | 'month' | 'year' | 'inception'

export const BUCKET_ORDER: BucketKey[] = ['today', 'week', 'month', 'year', 'inception']

export interface PerformanceBucket {
  key: BucketKey
  label: string
  start_day: string | null
  start_equity: number
  end_equity: number
  pnl: number
  pnl_pct: number
  sessions: number
}

export interface EquityPoint {
  day: string
  equity: number
  profit_loss: number
  profit_loss_pct: number
}

export interface AccountSummary {
  account_number?: string
  status?: string
  currency?: string
  equity?: number
  last_equity?: number
  cash?: number
  buying_power?: number
  long_market_value?: number
  multiplier?: number
  pattern_day_trader?: boolean
  trading_blocked?: boolean
  account_blocked?: boolean
  created_at?: string
}

export interface Position {
  symbol: string
  qty?: number
  side?: string
  avg_entry_price?: number
  current_price?: number
  market_value?: number
  cost_basis?: number
  unrealized_pl?: number
  unrealized_plpc?: number
  change_today?: number
}

export interface ClosedTrade {
  symbol: string
  qty: number
  entry_price: number
  exit_price: number
  entry_notional: number
  exit_notional: number
  pnl: number
  pnl_pct: number
}

export interface BasketTotals {
  entry_notional?: number
  exit_notional?: number
  pnl?: number
  pnl_pct?: number
}

export interface StrategyState {
  status: string | null
  entry_date: string | null
  exit_date: string | null
  symbols: string[]
  filled_symbols: string[]
  remaining_symbols: string[]
  share_mode?: 'whole' | 'fractional' | null
  budget: number
  per_symbol_notional: number
  estimated_deployed_notional?: number
  target_quantities?: Record<string, number>
  skipped_symbols?: string[]
  entry_completed_at: string | null
  exit_completed_at: string | null
  ranking_trade_date: string | null
  ranking_completed_at: string | null
  updated_at: string | null
}

export interface Statistics {
  max_drawdown: number
  max_drawdown_pct: number
  best_day: EquityPoint | null
  worst_day: EquityPoint | null
  win_rate: number
  winning_sessions: number
  losing_sessions: number
  sessions: number
}

export interface MarketClock {
  is_open?: boolean
  next_open?: string
  next_close?: string
  timestamp?: string
}

export interface Snapshot {
  version: number
  updated_at: string
  trading_day: string
  account: AccountSummary
  performance: Record<BucketKey, PerformanceBucket>
  statistics: Statistics
  equity_curve: EquityPoint[]
  positions: Position[]
  strategy: StrategyState
  closed_basket: ClosedTrade[]
  basket_totals: BasketTotals
  market: MarketClock
  meta: Record<string, unknown>
}

export interface SessionRecord {
  trading_day: string
  last_action: string | null
  updated_at: string | null
  status: string | null
  entry_date: string | null
  exit_date: string | null
  symbols: string[]
  entry_notional: number
  exit_notional: number
  realized_pnl: number | null
  realized_return: number | null
  trades: ClosedTrade[]
  error: string | null
}
