import { describe, expect, it, vi } from 'vitest'
import { createDemoRepository } from '~/repositories/demo.adapter'

// A tiny in-memory stand-in for the Firestore SDK, mirroring the reference project's
// approach: no emulator, no network.
const store = {
  document: null as unknown,
  sessions: [] as unknown[]
}

vi.mock('firebase/firestore', () => ({
  doc: (_firestore: unknown, collection: string, id: string) => ({ path: `${collection}/${id}` }),
  collection: (parent: { path: string }, name: string) => ({ path: `${parent.path}/${name}` }),
  getDoc: async () => ({
    exists: () => store.document !== null,
    data: () => store.document
  }),
  getDocs: async () => ({
    docs: store.sessions.map(data => ({ data: () => data }))
  }),
  query: (reference: unknown) => reference,
  orderBy: () => 'orderBy',
  limit: () => 'limit'
}))

const { createFirestoreRepository } = await import('~/repositories/firestore.adapter')

describe('firestore adapter', () => {
  const repository = createFirestoreRepository({} as never, { collection: 'accounts', document: 'paper' })

  it('reports its kind', () => {
    expect(repository.kind).toBe('firestore')
  })

  it('returns null when the publisher has not run yet', async () => {
    store.document = null
    expect(await repository.snapshot()).toBeNull()
  })

  it('returns the published snapshot', async () => {
    store.document = { version: 1, trading_day: '2026-08-24' }
    expect(await repository.snapshot()).toMatchObject({ trading_day: '2026-08-24' })
  })

  it('reads the sessions subcollection', async () => {
    store.sessions = [{ trading_day: '2026-08-24' }, { trading_day: '2026-08-21' }]
    const sessions = await repository.sessions(10)
    expect(sessions.map(session => session.trading_day)).toEqual(['2026-08-24', '2026-08-21'])
  })
})

describe('demo adapter', () => {
  it('produces a self-consistent snapshot', async () => {
    const snapshot = await createDemoRepository().snapshot()
    expect(snapshot).not.toBeNull()
    expect(snapshot!.equity_curve.length).toBeGreaterThan(100)
    expect(Object.keys(snapshot!.performance).sort()).toEqual(
      ['inception', 'month', 'today', 'week', 'year']
    )
    // Curve days are unique and ascending, as the publisher guarantees.
    const days = snapshot!.equity_curve.map(point => point.day)
    expect(new Set(days).size).toBe(days.length)
    expect([...days].sort()).toEqual(days)
  })

  it('is deterministic between calls, so screenshots do not drift', async () => {
    const first = await createDemoRepository().snapshot()
    const second = await createDemoRepository().snapshot()
    expect(first!.equity_curve).toEqual(second!.equity_curve)
  })

  it('honours the session limit', async () => {
    expect((await createDemoRepository().sessions(5))).toHaveLength(5)
  })
})
