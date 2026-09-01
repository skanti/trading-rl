import { describe, expect, it } from 'vitest'
import {
  formatAxisCurrency,
  formatCompactCurrency,
  formatCurrency,
  formatDay,
  formatPercent,
  formatQuantity,
  formatRelative,
  formatSignedCurrency,
  formatSignedPercent,
  profitFactorToneClass,
  toneClass
} from '~/utils/format'

describe('currency', () => {
  it('formats and signs amounts', () => {
    expect(formatCurrency(1234.5)).toBe('$1,234.50')
    expect(formatSignedCurrency(1234.5)).toBe('+$1,234.50')
    expect(formatSignedCurrency(-1234.5)).toBe('-$1,234.50')
    expect(formatSignedCurrency(0)).toBe('+$0.00')
  })

  it('falls back on missing values rather than printing NaN', () => {
    expect(formatCurrency(undefined)).toBe('—')
    expect(formatSignedCurrency(null)).toBe('—')
    expect(formatCompactCurrency(Number.NaN)).toBe('—')
  })

  it('compacts axis labels', () => {
    expect(formatCompactCurrency(105000)).toBe('$105K')
  })

  it('keeps nearby chart ticks distinct inside a narrow domain', () => {
    expect(formatAxisCurrency(99967, 50)).toBe('$99,967')
    expect(formatAxisCurrency(99967.25, 5)).toBe('$99,967.25')
    expect(formatAxisCurrency(105000, 20000)).toBe('$105K')
  })
})

describe('percentages', () => {
  it('treats the input as a ratio', () => {
    expect(formatPercent(0.0125)).toBe('1.25%')
    expect(formatSignedPercent(0.0125)).toBe('+1.25%')
    expect(formatSignedPercent(-0.0125)).toBe('-1.25%')
  })
})

describe('quantities', () => {
  it('trims the trailing zeros fractional orders produce', () => {
    expect(formatQuantity(52.31)).toBe('52.31')
    expect(formatQuantity(10)).toBe('10')
    expect(formatQuantity(0)).toBe('0')
  })
})

describe('toneClass', () => {
  it('keeps a flat account neutral rather than green', () => {
    expect(toneClass(0)).toBe('text-slate-400')
    expect(toneClass(undefined)).toBe('text-slate-400')
    expect(toneClass(1)).toBe('text-emerald-400')
    expect(toneClass(-1)).toBe('text-rose-400')
  })
})

describe('profitFactorToneClass', () => {
  it('uses 1.0 as the break-even threshold', () => {
    expect(profitFactorToneClass(undefined)).toBe('text-slate-400')
    expect(profitFactorToneClass(1)).toBe('text-slate-400')
    expect(profitFactorToneClass(0.99)).toBe('text-rose-400')
    expect(profitFactorToneClass(1.01)).toBe('text-emerald-400')
    expect(profitFactorToneClass(Number.POSITIVE_INFINITY)).toBe('text-emerald-400')
  })
})

describe('dates', () => {
  it('formats a session day', () => {
    expect(formatDay('2026-08-24')).toBe('Aug 24, 2026')
    expect(formatDay(null)).toBe('—')
  })

  it('reports staleness so an unattended publisher is obvious', () => {
    const now = new Date('2026-08-25T12:00:00Z')
    expect(formatRelative('2026-08-25T11:59:30Z', now)).toBe('just now')
    expect(formatRelative('2026-08-25T11:55:00Z', now)).toBe('5 minutes ago')
    expect(formatRelative('2026-08-25T09:00:00Z', now)).toBe('3 hours ago')
    expect(formatRelative('2026-08-23T12:00:00Z', now)).toBe('2 days ago')
    expect(formatRelative(null, now)).toBe('never')
  })
})
