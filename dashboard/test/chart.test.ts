import { describe, expect, it } from 'vitest'
import { areaPath, buildScale, gridValues, linePath, tickIndices } from '~/utils/chart'
import type { EquityPoint } from '~/types/dashboard'

const GEOMETRY = { width: 900, height: 300, padding: { top: 10, right: 10, bottom: 30, left: 60 } }

function points(values: number[]): EquityPoint[] {
  return values.map((equity, index) => ({
    day: `2026-08-${String(index + 1).padStart(2, '0')}`,
    equity,
    profit_loss: 0,
    profit_loss_pct: 0
  }))
}

describe('buildScale', () => {
  it('maps the domain across the inner plot area', () => {
    const scale = buildScale(points([100, 200]), GEOMETRY)
    expect(scale.x(0)).toBe(60)
    expect(scale.x(1)).toBe(890) // padding.left + innerWidth (900 - 60 - 10)
    expect(scale.y(scale.max)).toBeCloseTo(10)
    expect(scale.y(scale.min)).toBeCloseTo(270)
  })

  it('pads a flat series instead of dividing by a zero range', () => {
    const scale = buildScale(points([100000, 100000, 100000]), GEOMETRY)
    expect(scale.max).toBeGreaterThan(scale.min)
    const y = scale.y(100000)
    expect(Number.isFinite(y)).toBe(true)
    // The flat line lands in the middle of the plot.
    expect(y).toBeCloseTo((10 + 270) / 2, 5)
  })

  it('survives an empty series', () => {
    const scale = buildScale([], GEOMETRY)
    expect(Number.isFinite(scale.y(0))).toBe(true)
  })
})

describe('paths', () => {
  it('builds a polyline through every point', () => {
    const scale = buildScale(points([100, 110, 120]), GEOMETRY)
    const path = linePath(points([100, 110, 120]), scale)
    expect(path.startsWith('M ')).toBe(true)
    expect(path.match(/L /g)).toHaveLength(2)
  })

  it('draws a single point as a horizontal rule', () => {
    const single = points([100])
    const scale = buildScale(single, GEOMETRY)
    const path = linePath(single, scale)
    expect(path).toContain('M ')
    expect(path).toContain('L ')
  })

  it('returns an empty path for no data', () => {
    expect(linePath([], buildScale([], GEOMETRY))).toBe('')
    expect(areaPath([], buildScale([], GEOMETRY), 270)).toBe('')
  })

  it('closes the area back to the baseline', () => {
    const data = points([100, 120])
    const path = areaPath(data, buildScale(data, GEOMETRY), 270)
    expect(path.endsWith('Z')).toBe(true)
  })
})

describe('ticks', () => {
  it('always ends on the most recent session', () => {
    const indices = tickIndices(100, 5)
    expect(indices[0]).toBe(0)
    expect(indices[indices.length - 1]).toBe(99)
  })

  it('returns every index for a short series', () => {
    expect(tickIndices(3, 5)).toEqual([0, 1, 2])
    expect(tickIndices(0)).toEqual([])
  })

  it('spans the domain with gridlines', () => {
    const scale = buildScale(points([100, 200]), GEOMETRY)
    const values = gridValues(scale, 4)
    expect(values).toHaveLength(5)
    expect(values[0]).toBeCloseTo(scale.min)
    expect(values[4]).toBeCloseTo(scale.max)
  })
})
