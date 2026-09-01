import type { EquityPoint } from '~/types/dashboard'

export type PerformancePeriod
  = 'this_week'
    | 'this_month'
    | 'this_year'
    | 'inception'
    | 'custom'

export interface PerformanceWindow {
  label: string
  start: string | null
  end: string
}

export interface PeriodMetrics {
  startDay: string | null
  endDay: string | null
  startEquity: number
  endEquity: number
  pnl: number
  totalReturn: number
  annualizedReturn: number | null
  averageReturn: number | null
  winRate: number | null
  profitFactor: number | null
  annualizedVolatility: number | null
  sharpe: number | null
  sortino: number | null
  maxDrawdown: number
  maxDrawdownPct: number
  bestSession: { day: string, value: number } | null
  worstSession: { day: string, value: number } | null
  sessions: number
}

const DAY_MS = 86_400_000

function parseDay(day: string): Date {
  return new Date(`${day}T00:00:00Z`)
}

function formatDay(day: Date): string {
  return day.toISOString().slice(0, 10)
}

function shiftDay(day: Date, amount: number): Date {
  return new Date(day.getTime() + amount * DAY_MS)
}

function monday(day: Date): Date {
  const weekday = day.getUTCDay() || 7
  return shiftDay(day, 1 - weekday)
}

export function performanceWindow(
  period: PerformancePeriod,
  asOf: string,
  customStart?: string | null,
  customEnd?: string | null
): PerformanceWindow {
  const today = parseDay(asOf)
  const thisMonday = monday(today)
  switch (period) {
    case 'this_week':
      return { label: 'This week', start: formatDay(thisMonday), end: asOf }
    case 'this_month':
      return {
        label: 'This month',
        start: `${asOf.slice(0, 7)}-01`,
        end: asOf
      }
    case 'this_year':
      return { label: 'This year', start: `${asOf.slice(0, 4)}-01-01`, end: asOf }
    case 'custom':
      return { label: 'Custom', start: customStart || asOf, end: customEnd || asOf }
    case 'inception':
      return { label: 'Since inception', start: null, end: asOf }
  }
}

function sampleDeviation(values: number[]): number | null {
  if (values.length < 2) return null
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0)
    / (values.length - 1)
  return Math.sqrt(variance)
}

/** Calculate realized-only statistics from consecutive closed-session equity points. */
export function periodMetrics(
  source: EquityPoint[],
  window: Pick<PerformanceWindow, 'start' | 'end'>
): PeriodMetrics {
  const points = [...source]
    .filter(point => Number.isFinite(point.equity) && point.equity > 0)
    .sort((left, right) => left.day.localeCompare(right.day))
  const endIndex = points.findLastIndex(point => point.day <= window.end)
  if (endIndex < 0) {
    return {
      startDay: window.start,
      endDay: null,
      startEquity: 0,
      endEquity: 0,
      pnl: 0,
      totalReturn: 0,
      annualizedReturn: null,
      averageReturn: null,
      winRate: null,
      profitFactor: null,
      annualizedVolatility: null,
      sharpe: null,
      sortino: null,
      maxDrawdown: 0,
      maxDrawdownPct: 0,
      bestSession: null,
      worstSession: null,
      sessions: 0
    }
  }

  let anchorIndex = 0
  if (window.start) {
    const prior = points.findLastIndex((point, index) => index <= endIndex && point.day < window.start!)
    anchorIndex = prior >= 0 ? prior : 0
  }

  const moves: Array<{ day: string, value: number }> = []
  for (let index = Math.max(1, anchorIndex + 1); index <= endIndex; index += 1) {
    const current = points[index]!
    const previous = points[index - 1]!
    if (window.start && current.day < window.start) continue
    moves.push({ day: current.day, value: current.equity / previous.equity - 1 })
  }

  const startEquity = points[anchorIndex]!.equity
  const endEquity = moves.length ? points[endIndex]!.equity : startEquity
  const returns = moves.map(move => move.value)
  const mean = returns.length
    ? returns.reduce((sum, value) => sum + value, 0) / returns.length
    : null
  const deviation = sampleDeviation(returns)
  const downside = returns.length
    ? Math.sqrt(returns.reduce((sum, value) => sum + Math.min(value, 0) ** 2, 0) / returns.length)
    : null
  const gains = returns.filter(value => value > 0).reduce((sum, value) => sum + value, 0)
  const losses = Math.abs(returns.filter(value => value < 0).reduce((sum, value) => sum + value, 0))

  let peak = startEquity
  let maxDrawdown = 0
  let maxDrawdownPct = 0
  for (let index = anchorIndex + 1; index <= endIndex; index += 1) {
    const equity = points[index]!.equity
    peak = Math.max(peak, equity)
    const drawdown = peak - equity
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown
      maxDrawdownPct = peak > 0 ? drawdown / peak : 0
    }
  }

  const totalReturn = startEquity > 0 ? endEquity / startEquity - 1 : 0
  const best = moves.length
    ? moves.reduce((result, move) => move.value > result.value ? move : result)
    : null
  const worst = moves.length
    ? moves.reduce((result, move) => move.value < result.value ? move : result)
    : null

  return {
    startDay: window.start ?? points[anchorIndex]!.day,
    endDay: moves.length ? moves[moves.length - 1]!.day : null,
    startEquity,
    endEquity,
    pnl: endEquity - startEquity,
    totalReturn,
    annualizedReturn: returns.length && 1 + totalReturn > 0
      ? (1 + totalReturn) ** (252 / returns.length) - 1
      : null,
    averageReturn: mean,
    winRate: returns.length
      ? returns.filter(value => value > 0).length / returns.length
      : null,
    profitFactor: returns.length
      ? losses > 0 ? gains / losses : gains > 0 ? Number.POSITIVE_INFINITY : null
      : null,
    annualizedVolatility: deviation === null ? null : deviation * Math.sqrt(252),
    sharpe: mean !== null && deviation !== null && deviation > 0
      ? Math.sqrt(252) * mean / deviation
      : null,
    sortino: mean !== null && downside !== null && downside > 0
      ? Math.sqrt(252) * mean / downside
      : null,
    maxDrawdown,
    maxDrawdownPct,
    bestSession: best,
    worstSession: worst,
    sessions: returns.length
  }
}
