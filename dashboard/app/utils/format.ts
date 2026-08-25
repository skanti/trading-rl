/** Presentation helpers. Pure, so they are unit tested directly. */

const CURRENCY = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
})

const COMPACT_CURRENCY = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1
})

/** Fractional notional orders produce long decimals; trim to something readable. */
export function formatQuantity(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  const text = (value as number).toFixed(6).replace(/\.?0+$/, '')
  return text.length ? text : '0'
}

export function formatCurrency(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  return CURRENCY.format(value as number)
}

export function formatCompactCurrency(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  return COMPACT_CURRENCY.format(value as number)
}

export function formatSignedCurrency(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  const amount = value as number
  return `${amount >= 0 ? '+' : '-'}${CURRENCY.format(Math.abs(amount))}`
}

/** Input is a ratio (0.0125), not an already-multiplied percentage. */
export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  return `${((value as number) * 100).toFixed(digits)}%`
}

export function formatSignedPercent(value: number | null | undefined, digits = 2): string {
  if (!Number.isFinite(value ?? NaN)) return '—'
  const ratio = value as number
  return `${ratio >= 0 ? '+' : '-'}${(Math.abs(ratio) * 100).toFixed(digits)}%`
}

/**
 * Tailwind text colour for a signed number. Zero stays neutral rather than green, so a
 * flat, untraded account does not read as a win.
 */
export function toneClass(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN) || value === 0) return 'text-slate-400'
  return (value as number) > 0 ? 'text-emerald-400' : 'text-rose-400'
}

export function formatDay(day: string | null | undefined): string {
  if (!day) return '—'
  const parsed = new Date(`${day}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) return day
  return parsed.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

/** "3 minutes ago" style staleness, so an unattended publisher is obvious. */
export function formatRelative(value: string | null | undefined, now: Date = new Date()): string {
  if (!value) return 'never'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return 'never'
  const seconds = Math.round((now.getTime() - parsed.getTime()) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`
  const days = Math.round(hours / 24)
  return `${days} day${days === 1 ? '' : 's'} ago`
}
