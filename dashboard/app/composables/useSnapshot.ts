import type { SessionRecord, Snapshot } from '~/types/dashboard'

/**
 * Shared account snapshot. Pages read the same state rather than each refetching, and
 * `refresh` is exposed so the header can force a reload after the publisher runs.
 */
export function useSnapshot() {
  const repository = useRepository()

  const snapshot = useState<Snapshot | null>('dashboard:snapshot', () => null)
  const sessions = useState<SessionRecord[]>('dashboard:sessions', () => [])
  const pending = useState<boolean>('dashboard:pending', () => false)
  const error = useState<string | null>('dashboard:error', () => null)
  const loaded = useState<boolean>('dashboard:loaded', () => false)

  async function refresh(): Promise<void> {
    pending.value = true
    error.value = null
    try {
      const [latest, history] = await Promise.all([
        repository.snapshot(),
        repository.sessions(60)
      ])
      snapshot.value = latest
      sessions.value = history
      loaded.value = true
    } catch (cause) {
      // A permission error here almost always means the signed-in user is not allowed
      // by firestore.rules; surface it rather than showing a silently empty dashboard.
      error.value = (cause as Error)?.message ?? 'Could not load the account snapshot.'
    } finally {
      pending.value = false
    }
  }

  async function ensureLoaded(): Promise<void> {
    if (loaded.value || pending.value) return
    await refresh()
  }

  return {
    snapshot,
    sessions,
    pending,
    error,
    loaded,
    refresh,
    ensureLoaded,
    source: repository.kind
  }
}
