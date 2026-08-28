import type { SnapshotRepository } from './types'
import type { EquityPoint, SessionRecord, Snapshot } from '~/types/dashboard'

/**
 * Offline stand-in used when Firebase is not configured, so `pnpm dev` and the test
 * suite work without credentials. The curve is generated from a fixed seed, which
 * keeps screenshots and snapshot tests stable between runs.
 */

function seeded(seed: number): () => number {
  let state = seed
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296
    return state / 4294967296
  }
}

function buildCurve(sessions: number): EquityPoint[] {
  const random = seeded(20260825)
  const points: EquityPoint[] = []
  let equity = 100000
  let previous = equity
  const day = new Date('2026-02-02T00:00:00Z')

  while (points.length < sessions) {
    const weekday = day.getUTCDay()
    if (weekday !== 0 && weekday !== 6) {
      // Small positive drift with realistic daily noise.
      equity *= 1 + (random() - 0.46) * 0.018
      points.push({
        day: day.toISOString().slice(0, 10),
        equity: Math.round(equity * 100) / 100,
        profit_loss: Math.round((equity - previous) * 100) / 100,
        profit_loss_pct: (equity - previous) / previous
      })
      previous = equity
    }
    day.setUTCDate(day.getUTCDate() + 1)
  }
  return points
}

function bucketFrom(
  key: Snapshot['performance'][keyof Snapshot['performance']]['key'],
  label: string,
  startEquity: number,
  endEquity: number,
  startDay: string,
  sessions: number
) {
  const pnl = endEquity - startEquity
  return {
    key,
    label,
    start_day: startDay,
    start_equity: startEquity,
    end_equity: endEquity,
    pnl,
    pnl_pct: startEquity ? pnl / startEquity : 0,
    sessions
  }
}

function buildSnapshot(): Snapshot {
  const curve = buildCurve(136)
  const last = curve[curve.length - 1]!
  const equity = Math.round(last.equity * 1.0021 * 100) / 100
  const at = (fromEnd: number) => curve[Math.max(0, curve.length - fromEnd)]!

  let peak = curve[0]!.equity
  let drawdown = 0
  let drawdownPct = 0
  for (const point of curve) {
    peak = Math.max(peak, point.equity)
    if (peak - point.equity > drawdown) {
      drawdown = peak - point.equity
      drawdownPct = drawdown / peak
    }
  }
  const winners = curve.filter(point => point.profit_loss > 0).length
  const losers = curve.filter(point => point.profit_loss < 0).length

  return {
    version: 1,
    updated_at: new Date().toISOString(),
    trading_day: last.day,
    account: {
      account_number: 'PA0DEMO0000',
      status: 'ACTIVE',
      currency: 'USD',
      equity,
      last_equity: last.equity,
      cash: 18422.55,
      buying_power: equity * 4,
      long_market_value: equity - 18422.55,
      multiplier: 4,
      created_at: '2026-02-02T14:30:00Z'
    },
    performance: {
      today: bucketFrom('today', 'Today', last.equity, equity, at(2).day, 1),
      week: bucketFrom('week', 'Week to date', at(3).equity, equity, at(3).day, 2),
      month: bucketFrom('month', 'Month to date', at(18).equity, equity, at(18).day, 17),
      year: bucketFrom('year', 'Year to date', 100000, equity, curve[0]!.day, curve.length),
      inception: bucketFrom('inception', 'Since inception', 100000, equity, curve[0]!.day, curve.length)
    },
    statistics: {
      max_drawdown: Math.round(drawdown * 100) / 100,
      max_drawdown_pct: drawdownPct,
      best_day: curve.reduce((best, point) => (point.profit_loss > best.profit_loss ? point : best)),
      worst_day: curve.reduce((worst, point) => (point.profit_loss < worst.profit_loss ? point : worst)),
      win_rate: winners / Math.max(1, winners + losers),
      winning_sessions: winners,
      losing_sessions: losers,
      sessions: curve.length
    },
    equity_curve: curve,
    positions: [
      { symbol: 'NVDA', qty: 52.31, side: 'long', avg_entry_price: 178.2, current_price: 180.05, market_value: 9418.4, cost_basis: 9321.64, unrealized_pl: 96.76, unrealized_plpc: 0.0104, change_today: 0.0104 },
      { symbol: 'TSLA', qty: 31.05, side: 'long', avg_entry_price: 302.11, current_price: 298.44, market_value: 9266.56, cost_basis: 9380.52, unrealized_pl: -113.96, unrealized_plpc: -0.0121, change_today: -0.0121 },
      { symbol: 'AAPL', qty: 44.9, side: 'long', avg_entry_price: 221.4, current_price: 223.1, market_value: 10017.19, cost_basis: 9940.86, unrealized_pl: 76.33, unrealized_plpc: 0.0077, change_today: 0.0077 },
      { symbol: 'MSFT', qty: 19.8, side: 'long', avg_entry_price: 502.66, current_price: 508.2, market_value: 10062.36, cost_basis: 9952.67, unrealized_pl: 109.69, unrealized_plpc: 0.011, change_today: 0.011 }
    ],
    strategy: {
      status: 'open',
      entry_date: at(2).day,
      exit_date: last.day,
      symbols: ['NVDA', 'TSLA', 'AAPL', 'MSFT'],
      filled_symbols: ['NVDA', 'TSLA', 'AAPL', 'MSFT'],
      remaining_symbols: [],
      budget: 38595.69,
      per_symbol_notional: 9648.92,
      entry_completed_at: `${at(2).day}T19:59:12Z`,
      exit_completed_at: null,
      ranking_trade_date: at(2).day,
      ranking_completed_at: `${at(2).day}T19:05:44Z`,
      updated_at: new Date().toISOString()
    },
    closed_basket: [],
    basket_totals: {},
    market: { is_open: false, next_open: `${last.day}T13:30:00Z` },
    meta: { trading_url: 'demo', paper: true }
  }
}

function buildSessions(snapshot: Snapshot): SessionRecord[] {
  const symbols = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD']
  return snapshot.equity_curve
    .slice(-24)
    .reverse()
    .map((point, index) => {
      const chosen = symbols.slice(0, 3 + (index % 3))
      const perSymbol = point.profit_loss / chosen.length
      return {
        trading_day: point.day,
        last_action: 'exit',
        updated_at: `${point.day}T13:35:00Z`,
        status: 'closed',
        entry_date: point.day,
        exit_date: point.day,
        symbols: chosen,
        entry_equity: null,
        exit_equity: null,
        entry_notional: 38000,
        exit_notional: 38000 + point.profit_loss,
        gross_realized_pnl: point.profit_loss,
        gross_realized_return: point.profit_loss_pct,
        realized_pnl: point.profit_loss,
        realized_return: point.profit_loss_pct,
        trades: chosen.map((symbol, position) => {
          const entry = 100 + position * 37.5
          const pnl = perSymbol
          const qty = 80
          return {
            symbol,
            qty,
            entry_price: entry,
            exit_price: Math.round((entry + pnl / qty) * 10000) / 10000,
            entry_notional: entry * qty,
            exit_notional: entry * qty + pnl,
            pnl,
            pnl_pct: pnl / (entry * qty)
          }
        }),
        error: null
      }
    })
}

export function createDemoRepository(): SnapshotRepository {
  const snapshot = buildSnapshot()
  const sessions = buildSessions(snapshot)
  return {
    kind: 'demo',
    snapshot: async () => snapshot,
    sessions: async (limit = 60) => sessions.slice(0, limit)
  }
}
