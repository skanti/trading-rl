import { describe, expect, it } from 'vitest'
import {
  buildNaiveSchedule,
  buildSessionTimeline,
  executionMilestone,
  formatCountdown,
  lastCompletedTimelineIndex,
  zonedDateTime,
  type ScheduleConfig
} from '~/utils/schedule'
import type { StrategyState } from '~/types/dashboard'

const config: ScheduleConfig = {
  timeZone: 'America/New_York',
  rankingTime: '14:00',
  entryTime: '15:45',
  exitTime: '08:00'
}

describe('zonedDateTime', () => {
  it('converts New York summer and winter clocks through DST', () => {
    expect(zonedDateTime('2026-08-27', '09:00', config.timeZone).toISOString())
      .toBe('2026-08-27T13:00:00.000Z')
    expect(zonedDateTime('2026-12-03', '09:00', config.timeZone).toISOString())
      .toBe('2026-12-03T14:00:00.000Z')
  })
})

describe('buildNaiveSchedule', () => {
  it('shows exit, ranking and entry while holding overnight', () => {
    const events = buildNaiveSchedule(
      new Date('2026-08-27T06:30:00Z'),
      { status: 'open', exit_date: '2026-08-27', ranking_trade_date: '2026-08-26' },
      config
    )

    expect(events.map(event => event.key)).toEqual(['exit', 'rank', 'entry'])
    expect(events.map(event => event.at.toISOString())).toEqual([
      '2026-08-27T12:00:00.000Z',
      '2026-08-27T18:00:00.000Z',
      '2026-08-27T19:45:00.000Z'
    ])
  })

  it('skips weekends for the next naive trading cycle', () => {
    const events = buildNaiveSchedule(
      new Date('2026-08-28T21:00:00Z'),
      { status: 'closed', exit_date: '2026-08-28', ranking_trade_date: '2026-08-28' },
      config
    )

    expect(events.map(event => event.at.toISOString())).toEqual([
      '2026-08-31T18:00:00.000Z',
      '2026-08-31T19:45:00.000Z',
      '2026-09-01T12:00:00.000Z'
    ])
  })
})

describe('buildSessionTimeline', () => {
  it('shows the complete Friday-to-Monday holding cycle over a weekend', () => {
    const timeline = buildSessionTimeline(
      new Date('2026-08-29T16:00:00Z'),
      { status: 'open', entry_date: '2026-08-28', exit_date: '2026-08-31' },
      config,
      {
        sessions: [
          { date: '2026-08-28', open: '09:30', close: '16:00' },
          { date: '2026-08-31', open: '09:30', close: '16:00' }
        ]
      }
    )

    expect(timeline?.events.map(event => event.key)).toEqual([
      'market_open', 'rank', 'entry', 'market_close', 'exit', 'next_open'
    ])
    expect(timeline?.events.map(event => event.at.toISOString())).toEqual([
      '2026-08-28T13:30:00.000Z',
      '2026-08-28T18:00:00.000Z',
      '2026-08-28T19:45:00.000Z',
      '2026-08-28T20:00:00.000Z',
      '2026-08-31T12:00:00.000Z',
      '2026-08-31T13:30:00.000Z'
    ])
    expect(timeline?.phase).toBe('Holding overnight')
    expect(timeline?.context).toBe('Weekend · position held')
    expect(timeline?.progress).toBeGreaterThan(60)
    expect(timeline?.progress).toBeLessThan(80)
  })

  it('uses the exchange calendar to skip a weekday holiday', () => {
    const timeline = buildSessionTimeline(
      new Date('2026-07-03T16:00:00Z'),
      { status: 'closed', entry_date: '2026-07-02', exit_date: '2026-07-03' },
      config,
      {
        sessions: [
          { date: '2026-07-02', open: '09:30', close: '16:00' },
          { date: '2026-07-06', open: '09:30', close: '16:00' },
          { date: '2026-07-07', open: '09:30', close: '16:00' }
        ]
      }
    )

    expect(timeline?.entryDate).toBe('2026-07-06')
    expect(timeline?.exitDate).toBe('2026-07-07')
    expect(timeline?.context).toBe('Market holiday')
    expect(timeline?.progress).toBe(0)
  })

  it('skips an early-close session that cannot fit the configured entry', () => {
    const timeline = buildSessionTimeline(
      new Date('2026-11-27T15:00:00Z'),
      { status: 'closed', entry_date: null, exit_date: null },
      config,
      {
        sessions: [
          { date: '2026-11-27', open: '09:30', close: '13:00' },
          { date: '2026-11-30', open: '09:30', close: '16:00' },
          { date: '2026-12-01', open: '09:30', close: '16:00' }
        ]
      }
    )

    expect(timeline?.entryDate).toBe('2026-11-30')
    expect(timeline?.context).toBe('Early close today at 13:00 · entry cycle skipped')
  })
})

describe('formatCountdown', () => {
  it('formats future and overdue durations compactly', () => {
    const now = new Date('2026-08-27T10:00:00Z')
    expect(formatCountdown(new Date('2026-08-27T13:05:00Z'), now)).toBe('in 3h 05m')
    expect(formatCountdown(new Date('2026-08-27T09:30:00Z'), now)).toBe('0h 30m overdue')
  })
})

describe('lastCompletedTimelineIndex', () => {
  const events = [
    { at: new Date('2026-08-27T13:30:00Z') },
    { at: new Date('2026-08-27T18:00:00Z') },
    { at: new Date('2026-08-27T19:45:00Z') }
  ]

  it('keeps the next unreached milestone neutral', () => {
    expect(lastCompletedTimelineIndex(events, new Date('2026-08-27T18:30:00Z'))).toBe(1)
  })

  it('has no active milestone before the schedule begins', () => {
    expect(lastCompletedTimelineIndex(events, new Date('2026-08-27T12:00:00Z'))).toBeUndefined()
  })
})

describe('executionMilestone', () => {
  const symbols = ['AAPL', 'MSFT']
  const strategy: StrategyState = {
    status: 'exit_queued',
    entry_date: '2026-09-08',
    exit_date: '2026-09-09',
    symbols,
    filled_symbols: symbols,
    remaining_symbols: symbols,
    budget: 10000,
    per_symbol_notional: 5000,
    entry_completed_at: '2026-09-08T19:45:01Z',
    exit_completed_at: null,
    ranking_trade_date: '2026-09-08',
    ranking_completed_at: '2026-09-08T18:01:00Z',
    updated_at: '2026-09-09T10:00:01Z'
  }
  const now = new Date('2026-09-09T12:00:00Z')
  const timeline = buildSessionTimeline(now, strategy, { ...config, exitTime: '06:00' }, {
    sessions: [
      { date: '2026-09-08', open: '09:30', close: '16:00' },
      { date: '2026-09-09', open: '09:30', close: '16:00' }
    ]
  })!
  const exit = timeline.events.find(event => event.key === 'exit')!
  const opening = timeline.events.find(event => event.key === 'next_open')!

  it('marks queued orders complete and leaves the next opening available for its countdown', () => {
    expect(executionMilestone(exit, strategy, now, config.timeZone)).toEqual({
      active: false, completed: true, warning: false, description: 'Queued successfully'
    })
    expect(timeline.events.some(event => executionMilestone(event, strategy, now, config.timeZone)?.active)).toBe(false)
    expect(timeline.nextEvent?.key).toBe('next_open')
    expect(formatCountdown(timeline.nextEvent!.at, now)).toBe('in 1h 30m')
  })

  it('waits for fill confirmation after the open instead of showing a completed exit', () => {
    expect(executionMilestone(opening, strategy, new Date('2026-09-09T13:30:01Z'), config.timeZone)).toEqual({
      active: true, completed: false, warning: false, description: 'Awaiting fills · 2 remaining'
    })
    const closed = { ...strategy, status: 'closed', remaining_symbols: [], exit_completed_at: '2026-09-09T13:30:02Z' }
    expect(executionMilestone(exit, closed, new Date('2026-09-09T13:31:00Z'), config.timeZone)?.description).toBe('Exited · 9:30 AM')
    expect(executionMilestone(opening, closed, new Date('2026-09-09T13:31:00Z'), config.timeZone)).toBeNull()
  })

  it('never labels unsent or failed exits as successfully queued', () => {
    expect(executionMilestone(exit, { ...strategy, status: 'open' }, now, config.timeZone)?.description).toBe('Awaiting exit orders · 2 remaining')
    expect(executionMilestone(exit, { ...strategy, status: 'exiting' }, now, config.timeZone)?.description).toBe('Exiting · 2 remaining')
    expect(executionMilestone(exit, { ...strategy, status: 'exit_incomplete' }, now, config.timeZone)).toEqual({
      active: true, completed: false, warning: true, description: 'Exit incomplete · 2 remaining'
    })
  })
})
