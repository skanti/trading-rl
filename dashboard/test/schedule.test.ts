import { describe, expect, it } from 'vitest'
import {
  buildNaiveSchedule,
  buildSessionTimeline,
  formatCountdown,
  lastCompletedTimelineIndex,
  zonedDateTime,
  type ScheduleConfig
} from '~/utils/schedule'

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
