import type { SessionRecord, Snapshot } from '~/types/dashboard'

/**
 * Read-only view of the published data. Two adapters implement it: Firestore for the
 * real account, and a demo adapter so `pnpm dev` works before Firebase is provisioned.
 */
export interface SnapshotRepository {
  kind: 'firestore' | 'demo'
  snapshot: () => Promise<Snapshot | null>
  sessions: (limit?: number) => Promise<SessionRecord[]>
}
