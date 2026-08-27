import { describe, expect, it } from 'vitest'
import {
  buildNaiveSchedule,
  formatCountdown,
  zonedDateTime,
  type ScheduleConfig
} from '~/utils/schedule'

const config: ScheduleConfig = {
  timeZone: 'America/New_York',
  rankingTime: '15:00',
  entryTime: '15:59',
  exitTime: '09:00'
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
      '2026-08-27T13:00:00.000Z',
      '2026-08-27T19:00:00.000Z',
      '2026-08-27T19:59:00.000Z'
    ])
  })

  it('skips weekends for the next naive trading cycle', () => {
    const events = buildNaiveSchedule(
      new Date('2026-08-28T21:00:00Z'),
      { status: 'closed', exit_date: '2026-08-28', ranking_trade_date: '2026-08-28' },
      config
    )

    expect(events.map(event => event.at.toISOString())).toEqual([
      '2026-08-31T19:00:00.000Z',
      '2026-08-31T19:59:00.000Z',
      '2026-09-01T13:00:00.000Z'
    ])
  })
})

describe('formatCountdown', () => {
  it('formats future and overdue durations compactly', () => {
    const now = new Date('2026-08-27T10:00:00Z')
    expect(formatCountdown(new Date('2026-08-27T13:05:00Z'), now)).toBe('in 3h 05m')
    expect(formatCountdown(new Date('2026-08-27T09:30:00Z'), now)).toBe('0h 30m overdue')
  })
})
