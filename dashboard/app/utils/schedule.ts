import type { MarketClock, MarketSession, StrategyState } from '~/types/dashboard'

export interface ScheduleConfig {
  timeZone: string
  rankingTime: string
  entryTime: string
  exitTime: string
}

export interface ScheduleEvent {
  key: 'market_open' | 'rank' | 'entry' | 'market_close' | 'exit' | 'next_open'
  label: string
  at: Date
}

export interface SessionTimeline {
  events: ScheduleEvent[]
  progress: number
  nextEvent: ScheduleEvent | null
  phase: string
  context: string
  entryDate: string
  exitDate: string
}

export interface ExecutionMilestone {
  active: boolean
  completed: boolean
  warning: boolean
  description?: string
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

function isActiveStrategy(strategy: Pick<StrategyState, 'status'>): boolean {
  return Boolean(
    strategy.status
    && strategy.status !== 'closed'
    && strategy.status !== 'entry_failed'
  )
}

function validSessions(sessions: MarketSession[]): MarketSession[] {
  const unique = new Map<string, MarketSession>()
  for (const session of sessions) {
    if (
      /^\d{4}-\d{2}-\d{2}$/.test(session.date)
      && /^\d{1,2}:\d{2}$/.test(session.open)
      && /^\d{1,2}:\d{2}$/.test(session.close)
    ) {
      unique.set(session.date, session)
    }
  }
  return [...unique.values()].sort((left, right) => left.date.localeCompare(right.date))
}

function completionTime(value: string | null, scheduledAt: Date): Date | null {
  if (!value) return null
  const completedAt = new Date(value)
  if (Number.isNaN(completedAt.getTime()) || completedAt < scheduledAt) return null
  return completedAt
}

/** Queue submission completes the exit-orders stage; fills remain a separate step. */
export function executionMilestone(
  event: ScheduleEvent,
  strategy: StrategyState,
  now: Date,
  timeZone: string
): ExecutionMilestone | null {
  if (event.key === 'entry') {
    const completedAt = completionTime(strategy.entry_completed_at, event.at)
    const total = strategy.symbols.length
    const filled = strategy.filled_symbols.length
    if (completedAt) {
      const warning = total > 0 && filled < total
      return {
        active: false,
        completed: true,
        warning,
        description: warning
          ? `${filled}/${total} filled · ${formatScheduleTime(completedAt, now, timeZone)}`
          : `Filled · ${formatScheduleTime(completedAt, now, timeZone)}`
      }
    }
    if (event.at <= now) {
      return {
        active: true,
        completed: false,
        warning: false,
        description: total > 0 ? `Filling · ${filled}/${total}` : 'Filling'
      }
    }
  }

  if (event.key === 'exit') {
    const completedAt = completionTime(strategy.exit_completed_at, event.at)
    const remaining = strategy.remaining_symbols.length
    if (completedAt) {
      return {
        active: false,
        completed: true,
        warning: remaining > 0,
        description: remaining > 0
          ? `${remaining} remaining · ${formatScheduleTime(completedAt, now, timeZone)}`
          : `Exited · ${formatScheduleTime(completedAt, now, timeZone)}`
      }
    }
    if (strategy.status === 'exit_queued') {
      return { active: false, completed: true, warning: false, description: 'Queued successfully' }
    }
    if (event.at <= now) {
      const label = strategy.status === 'exit_incomplete'
        ? 'Exit incomplete'
        : strategy.status === 'exiting' ? 'Exiting' : 'Awaiting exit orders'
      return {
        active: true,
        completed: false,
        warning: strategy.status === 'exit_incomplete',
        description: remaining > 0 ? `${label} · ${remaining} remaining` : label
      }
    }
  }

  if (event.key === 'next_open' && event.at <= now && strategy.status === 'exit_queued' && !strategy.exit_completed_at) {
    const remaining = strategy.remaining_symbols.length
    return {
      active: true,
      completed: false,
      warning: false,
      description: remaining > 0 ? `Awaiting fills · ${remaining} remaining` : 'Awaiting fill confirmation'
    }
  }
  return null
}

function sessionCanEnter(session: MarketSession, config: ScheduleConfig): boolean {
  const [openHour, openMinute] = clockParts(session.open)
  const [closeHour, closeMinute] = clockParts(session.close)
  const [rankHour, rankMinute] = clockParts(config.rankingTime)
  const [entryHour, entryMinute] = clockParts(config.entryTime)
  const open = openHour * 60 + openMinute
  const close = closeHour * 60 + closeMinute
  const rank = rankHour * 60 + rankMinute
  const entry = entryHour * 60 + entryMinute
  return rank >= open && rank < entry && entry < close
}

function stagedProgress(events: ScheduleEvent[], now: Date): number {
  if (events.length < 2 || now <= events[0]!.at) return 0
  if (now >= events[events.length - 1]!.at) return 100
  const nextIndex = events.findIndex(event => event.at > now)
  const previous = events[nextIndex - 1]!
  const next = events[nextIndex]!
  const duration = Math.max(1, next.at.getTime() - previous.at.getTime())
  const withinStage = (now.getTime() - previous.at.getTime()) / duration
  return ((nextIndex - 1 + withinStage) / (events.length - 1)) * 100
}

function nonTradingContext(
  now: Date,
  sessions: MarketSession[],
  selected: MarketSession,
  config: ScheduleConfig,
  active: boolean
): string {
  const local = zonedParts(now, config.timeZone)
  const today = isoDay(local)
  const todaySession = sessions.find(session => session.date === today)
  const hold = active ? ' · position held' : ''
  if (!todaySession) {
    return `${WEEKDAYS.has(local.weekday) ? 'Market holiday' : 'Weekend'}${hold}`
  }
  if (!active && !sessionCanEnter(todaySession, config) && selected.date !== today) {
    return `Early close today at ${todaySession.close} · entry cycle skipped`
  }
  const open = zonedDateTime(today, todaySession.open, config.timeZone)
  const close = zonedDateTime(today, todaySession.close, config.timeZone)
  if (now < open) return `Pre-market${hold}`
  if (now >= close) return `After hours${hold}`
  return active ? 'Regular session · position held' : 'Regular session'
}

/**
 * Build the complete overnight cycle from the exchange's real session calendar.
 * Early closes that cannot fit ranking and entry are skipped. The progress bar
 * gives every operational stage equal space, then interpolates by wall time inside
 * the current stage so ranking and entry remain visible beside the overnight hold.
 */
export function buildSessionTimeline(
  now: Date,
  strategy: Pick<StrategyState, 'status' | 'entry_date' | 'exit_date'>,
  config: ScheduleConfig,
  market: Pick<MarketClock, 'sessions'>
): SessionTimeline | null {
  const sessions = validSessions(market.sessions ?? [])
  if (sessions.length < 2) return null

  const active = isActiveStrategy(strategy)
  let entryIndex = active && strategy.entry_date
    ? sessions.findIndex(session => session.date === strategy.entry_date)
    : -1

  if (entryIndex < 0) {
    entryIndex = sessions.findIndex((session, index) => (
      index + 1 < sessions.length
      && sessionCanEnter(session, config)
      && now < zonedDateTime(session.date, config.entryTime, config.timeZone)
    ))
  }
  if (entryIndex < 0 || entryIndex + 1 >= sessions.length) return null

  const entrySession = sessions[entryIndex]!
  const configuredExit = strategy.exit_date
    ? sessions.find(session => session.date === strategy.exit_date)
    : undefined
  const exitSession = active && configuredExit
    ? configuredExit
    : sessions[entryIndex + 1]!

  const events: ScheduleEvent[] = [
    {
      key: 'market_open',
      label: 'Market opens',
      at: zonedDateTime(entrySession.date, entrySession.open, config.timeZone)
    },
    {
      key: 'rank',
      label: 'Rank stocks',
      at: zonedDateTime(entrySession.date, config.rankingTime, config.timeZone)
    },
    {
      key: 'entry',
      label: 'Open position',
      at: zonedDateTime(entrySession.date, config.entryTime, config.timeZone)
    },
    {
      key: 'market_close',
      label: 'Market closes',
      at: zonedDateTime(entrySession.date, entrySession.close, config.timeZone)
    },
    {
      key: 'exit',
      label: 'Exit orders',
      at: zonedDateTime(exitSession.date, config.exitTime, config.timeZone)
    },
    {
      key: 'next_open',
      label: 'Next market open',
      at: zonedDateTime(exitSession.date, exitSession.open, config.timeZone)
    }
  ]
  events.sort((left, right) => left.at.getTime() - right.at.getTime())

  const nextEvent = events.find(event => event.at > now) ?? null
  const previousEvent = [...events].reverse().find(event => event.at <= now)
  let phase = 'Waiting for the next trading session'
  if (!nextEvent) phase = 'Cycle complete'
  else if (previousEvent?.key === 'market_close') phase = 'Holding overnight'
  else if (previousEvent?.key === 'exit') phase = 'Exit window · awaiting market open'
  else if (previousEvent?.key === 'next_open') phase = 'Market open · waiting to exit'
  else if (previousEvent?.key === 'entry') phase = 'Position open'
  else if (previousEvent?.key === 'rank') phase = 'Ranking window passed · waiting to enter'
  else if (previousEvent?.key === 'market_open') phase = 'Market open · waiting to rank'

  return {
    events,
    progress: stagedProgress(events, now),
    nextEvent,
    phase,
    context: nonTradingContext(now, sessions, entrySession, config, active),
    entryDate: entrySession.date,
    exitDate: exitSession.date
  }
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
  const active = isActiveStrategy(strategy)
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

/** UTimeline uses a zero-based active index; point at the last reached event. */
export function lastCompletedTimelineIndex(
  events: Array<Pick<ScheduleEvent, 'at'>>,
  now: Date
): number | undefined {
  const completed = events.filter(event => event.at <= now).length
  return completed > 0 ? completed - 1 : undefined
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

export function formatScheduleDateTime(target: Date, now: Date, timeZone: string): string {
  const targetParts = zonedParts(target, timeZone)
  const today = isoDay(zonedParts(now, timeZone))
  const targetDay = isoDay(targetParts)
  const dateLabel = targetDay === today
    ? 'Today'
    : new Intl.DateTimeFormat('en-US', {
        timeZone,
        weekday: 'short',
        month: 'short',
        day: 'numeric'
      }).format(target)
  const clock = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit'
  }).format(target)
  return `${dateLabel} · ${clock}`
}

export function formatZonedNow(now: Date, timeZone: string): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short'
  }).format(now)
}
