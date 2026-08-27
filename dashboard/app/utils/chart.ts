import type { EquityPoint } from '~/types/dashboard'

export interface ChartGeometry {
  width: number
  height: number
  padding: { top: number, right: number, bottom: number, left: number }
}

export interface ChartScale {
  x: (index: number) => number
  y: (value: number) => number
  min: number
  max: number
  innerWidth: number
  innerHeight: number
}

/**
 * Build linear scales for the equity curve.
 *
 * A flat account -- which is exactly how this one starts -- has zero range, and naive
 * scaling would divide by zero and collapse the line onto an edge. Padding the domain
 * in that case draws the flat line through the middle of the plot instead.
 */
export function buildScale(
  points: EquityPoint[],
  geometry: ChartGeometry,
  domainPaddingRatio = 0.12
): ChartScale {
  const { width, height, padding } = geometry
  const innerWidth = Math.max(1, width - padding.left - padding.right)
  const innerHeight = Math.max(1, height - padding.top - padding.bottom)

  const values = points.map(point => point.equity).filter(value => Number.isFinite(value))
  let min = values.length ? Math.min(...values) : 0
  let max = values.length ? Math.max(...values) : 1

  if (max - min < Number.EPSILON) {
    const cushion = Math.max(Math.abs(max) * 0.01, 1)
    min -= cushion
    max += cushion
  } else {
    // Breathing room so the extremes do not sit exactly on the frame.
    const cushion = (max - min) * Math.max(0, domainPaddingRatio)
    min -= cushion
    max += cushion
  }

  const span = Math.max(1, points.length - 1)

  return {
    x: index => padding.left + (index / span) * innerWidth,
    y: value => padding.top + innerHeight - ((value - min) / (max - min)) * innerHeight,
    min,
    max,
    innerWidth,
    innerHeight
  }
}

/** `d` attribute for the equity line. */
export function linePath(points: EquityPoint[], scale: ChartScale): string {
  if (!points.length) return ''
  if (points.length === 1) {
    // One point is a horizontal rule rather than an invisible zero-length path.
    const y = scale.y(points[0]!.equity)
    return `M ${scale.x(0)} ${y} L ${scale.x(0) + scale.innerWidth} ${y}`
  }
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${scale.x(index)} ${scale.y(point.equity)}`)
    .join(' ')
}

/** `d` attribute for the filled area beneath the line. */
export function areaPath(points: EquityPoint[], scale: ChartScale, baseY: number): string {
  const line = linePath(points, scale)
  if (!line) return ''
  const lastX = points.length === 1 ? scale.x(0) + scale.innerWidth : scale.x(points.length - 1)
  return `${line} L ${lastX} ${baseY} L ${scale.x(0)} ${baseY} Z`
}

/**
 * Pick at most `count` evenly spaced indices to label, always including the last so the
 * axis ends on the most recent session.
 */
export function tickIndices(length: number, count = 5): number[] {
  if (length <= 0) return []
  if (length <= count) return Array.from({ length }, (_, index) => index)
  const step = (length - 1) / (count - 1)
  const indices = Array.from({ length: count }, (_, index) => Math.round(index * step))
  return Array.from(new Set(indices))
}

/** Evenly spaced horizontal gridline values across the scale's domain. */
export function gridValues(scale: ChartScale, count = 4): number[] {
  const step = (scale.max - scale.min) / count
  return Array.from({ length: count + 1 }, (_, index) => scale.min + index * step)
}
