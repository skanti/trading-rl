import { describe, expect, it } from 'vitest'
import { performanceWindow, periodMetrics } from '~/utils/performance'
import type { EquityPoint } from '~/types/dashboard'

const curve: EquityPoint[] = [
  { day: '2025-12-31', equity: 100, profit_loss: 0, profit_loss_pct: 0 },
  { day: '2026-01-02', equity: 110, profit_loss: 10, profit_loss_pct: 0.10 },
  { day: '2026-01-05', equity: 99, profit_loss: -1, profit_loss_pct: -0.01 },
  { day: '2026-01-06', equity: 118.8, profit_loss: 18.8, profit_loss_pct: 0.188 }
]

describe('performanceWindow', () => {
  it('builds the week-to-date calendar boundary', () => {
    expect(performanceWindow('this_week', '2026-01-07')).toMatchObject({
      start: '2026-01-05', end: '2026-01-07'
    })
  })

  it('uses both selected days as custom inclusive boundaries', () => {
    expect(performanceWindow('custom', '2026-01-07', '2026-01-02', '2026-01-06')).toMatchObject({
      start: '2026-01-02', end: '2026-01-06'
    })
  })
})

describe('periodMetrics', () => {
  it('calculates risk and return metrics from consecutive session equity', () => {
    const metrics = periodMetrics(curve, { start: '2026-01-05', end: '2026-01-07' })

    expect(metrics.sessions).toBe(2)
    expect(metrics.startEquity).toBe(110)
    expect(metrics.endEquity).toBe(118.8)
    expect(metrics.pnl).toBeCloseTo(8.8)
    expect(metrics.totalReturn).toBeCloseTo(0.08)
    expect(metrics.averageReturn).toBeCloseTo(0.05)
    expect(metrics.winRate).toBe(0.5)
    expect(metrics.profitFactor).toBeCloseTo(2)
    expect(metrics.maxDrawdown).toBeCloseTo(11)
    expect(metrics.maxDrawdownPct).toBeCloseTo(0.1)
    expect(metrics.bestSession?.day).toBe('2026-01-06')
    expect(metrics.bestSession?.value).toBeCloseTo(0.2)
    expect(metrics.worstSession?.day).toBe('2026-01-05')
    expect(metrics.worstSession?.value).toBeCloseTo(-0.1)
    expect(metrics.sharpe).not.toBeNull()
    expect(metrics.sortino).not.toBeNull()
  })

  it('returns an honest empty period instead of NaN statistics', () => {
    const metrics = periodMetrics(curve, { start: '2026-02-01', end: '2026-02-28' })

    expect(metrics.sessions).toBe(0)
    expect(metrics.pnl).toBe(0)
    expect(metrics.sharpe).toBeNull()
    expect(metrics.winRate).toBeNull()
  })
})
