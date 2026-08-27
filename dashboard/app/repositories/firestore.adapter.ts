import {
  type Firestore,
  collection,
  doc,
  getDoc,
  getDocs,
  limit as limitTo,
  orderBy,
  query
} from 'firebase/firestore/lite'
import type { SnapshotRepository } from './types'
import type { SessionRecord, Snapshot } from '~/types/dashboard'

export interface FirestoreLocation {
  collection: string
  document: string
}

/**
 * Reads what `scripts/dashboard_daemon.py` writes. Nothing here writes: the rules in
 * firestore.rules reject browser writes outright.
 */
export function createFirestoreRepository(
  firestore: Firestore,
  location: FirestoreLocation
): SnapshotRepository {
  const accountReference = doc(firestore, location.collection, location.document)

  return {
    kind: 'firestore',

    async snapshot(): Promise<Snapshot | null> {
      const result = await getDoc(accountReference)
      // A missing document means the publisher has not run yet, which the UI shows as
      // an empty state rather than an error.
      return result.exists() ? (result.data() as Snapshot) : null
    },

    async sessions(limit = 60): Promise<SessionRecord[]> {
      const sessions = collection(accountReference, 'sessions')
      const result = await getDocs(
        query(sessions, orderBy('trading_day', 'desc'), limitTo(limit))
      )
      return result.docs.map(entry => entry.data() as SessionRecord)
    }
  }
}
