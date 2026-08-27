import type { StrategyState } from '~/types/dashboard'

export interface ScheduleConfig {
  timeZone: string
  rankingTime: string
  entryTime: string
  exitTime: string
}

export interface ScheduleEvent {
  key: 'rank' | 'entry' | 'exit'
  label: string
  at: Date
}

interface ZonedParts {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  weekday: string
}

const WEEKDAYS = new Set(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])

function clockParts(clock: string): [number, number] {
  const match = /^(\d{1,2}):(\d{2})$/.exec(clock)
  if (!match) throw new Error(`invalid schedule clock: ${clock}`)
  const hour = Number(match[1])
  const minute = Number(match[2])
  if (hour > 23 || minute > 59) throw new Error(`invalid schedule clock: ${clock}`)
  return [hour, minute]
}

function zonedParts(value: Date, timeZone: string): ZonedParts {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    weekday: 'short'
  }).formatToParts(value)
  const part = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find(candidate => candidate.type === type)?.value ?? ''
  return {
    year: Number(part('year')),
    month: Number(part('month')),
    day: Number(part('day')),
    hour: Number(part('hour')),
    minute: Number(part('minute')),
    weekday: part('weekday')
  }
}

function isoDay(parts: Pick<ZonedParts, 'year' | 'month' | 'day'>): string {
  return `${parts.year}-${String(parts.month).padStart(2, '0')}-${String(parts.day).padStart(2, '0')}`
}

function shiftDay(day: string, amount: number): string {
  const [year, month, date] = day.split('-').map(Number)
  const shifted = new Date(Date.UTC(year!, month! - 1, date! + amount))
  return shifted.toISOString().slice(0, 10)
}

function weekdayFor(day: string): string {
  const [year, month, date] = day.split('-').map(Number)
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'UTC',
    weekday: 'short'
  }).format(new Date(Date.UTC(year!, month! - 1, date!)))
}

function nextWeekday(day: string, includeCurrent = false): string {
  let candidate = includeCurrent ? day : shiftDay(day, 1)
  while (!WEEKDAYS.has(weekdayFor(candidate))) candidate = shiftDay(candidate, 1)
  return candidate
}

/** Convert a wall-clock time in an IANA zone to a real timestamp without a date library. */
export function zonedDateTime(day: string, clock: string, timeZone: string): Date {
  const [year, month, date] = day.split('-').map(Number)
  const [hour, minute] = clockParts(clock)
  const targetAsUtc = Date.UTC(year!, month! - 1, date!, hour, minute)
  let timestamp = targetAsUtc
  // Re-evaluate the zone offset after each correction. Two passes cover DST offsets
  // around the dates used by the US equity calendar.
  for (let pass = 0; pass < 2; pass += 1) {
    const actual = zonedParts(new Date(timestamp), timeZone)
    const actualAsUtc = Date.UTC(
      actual.year,
      actual.month - 1,
      actual.day,
      actual.hour,
      actual.minute
    )
    timestamp += targetAsUtc - actualAsUtc
  }
  return new Date(timestamp)
}

function nextEntryDay(
  now: Date,
  strategy: Pick<StrategyState, 'ranking_trade_date'>,
  config: ScheduleConfig
): { day: string, rankingPending: boolean } {
  const local = zonedParts(now, config.timeZone)
  const today = isoDay(local)
  const weekday = WEEKDAYS.has(local.weekday)
  const rankingAt = zonedDateTime(today, config.rankingTime, config.timeZone)
  const entryAt = zonedDateTime(today, config.entryTime, config.timeZone)
  const rankingComplete = strategy.ranking_trade_date === today
  const todayIsViable = weekday && now < entryAt && (now < rankingAt || rankingComplete)
  return todayIsViable
    ? { day: today, rankingPending: !rankingComplete }
    : { day: nextWeekday(today), rankingPending: true }
}

/** Upcoming algorithm actions. Weekends are skipped; exchange holidays are intentionally not. */
export function buildNaiveSchedule(
  now: Date,
  strategy: Pick<StrategyState, 'status' | 'exit_date' | 'ranking_trade_date'>,
  config: ScheduleConfig
): ScheduleEvent[] {
  const events: ScheduleEvent[] = []
  const active = strategy.status && strategy.status !== 'closed' && strategy.status !== 'entry_failed'
  if (active && strategy.exit_date) {
    events.push({
      key: 'exit',
      label: 'Sell orders',
      at: zonedDateTime(strategy.exit_date, config.exitTime, config.timeZone)
    })
  }

  const entry = nextEntryDay(now, strategy, config)
  if (entry.rankingPending) {
    events.push({
      key: 'rank',
      label: 'Rank stocks',
      at: zonedDateTime(entry.day, config.rankingTime, config.timeZone)
    })
  }
  events.push({
    key: 'entry',
    label: 'Open position',
    at: zonedDateTime(entry.day, config.entryTime, config.timeZone)
  })

  if (!active) {
    const exitDay = nextWeekday(entry.day)
    events.push({
      key: 'exit',
      label: 'Sell orders',
      at: zonedDateTime(exitDay, config.exitTime, config.timeZone)
    })
  }

  return events.sort((left, right) => left.at.getTime() - right.at.getTime()).slice(0, 3)
}

export function formatCountdown(target: Date, now: Date): string {
  const totalMinutes = Math.round((target.getTime() - now.getTime()) / 60_000)
  if (Math.abs(totalMinutes) < 1) return 'now'
  const overdue = totalMinutes < 0
  const absolute = Math.abs(totalMinutes)
  const days = Math.floor(absolute / (24 * 60))
  const hours = Math.floor((absolute % (24 * 60)) / 60)
  const minutes = absolute % 60
  const duration = days
    ? `${days}d ${hours}h ${String(minutes).padStart(2, '0')}m`
    : `${hours}h ${String(minutes).padStart(2, '0')}m`
  return overdue ? `${duration} overdue` : `in ${duration}`
}

export function formatScheduleTime(target: Date, now: Date, timeZone: string): string {
  const targetParts = zonedParts(target, timeZone)
  const today = isoDay(zonedParts(now, timeZone))
  const targetDay = isoDay(targetParts)
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    ...(targetDay === today ? {} : { weekday: 'short' as const }),
    hour: 'numeric',
    minute: '2-digit'
  }).format(target)
}

export function formatZonedNow(now: Date, timeZone: string): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short'
  }).format(now)
}
