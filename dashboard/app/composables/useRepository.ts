import { createDemoRepository } from '~/repositories/demo.adapter'
import { createFirestoreRepository } from '~/repositories/firestore.adapter'
import type { SnapshotRepository } from '~/repositories/types'
import type { FirebaseServices } from '~/plugins/firebase.client'

/**
 * Firestore when Firebase is configured, the demo dataset otherwise. Mirrors the
 * fallback pattern used in the reference project so the app is never unusable.
 */
export function useRepository(): SnapshotRepository {
  const { $firebase } = useNuxtApp() as unknown as { $firebase: FirebaseServices | null }
  const config = useRuntimeConfig().public

  if (!$firebase) return createDemoRepository()

  return createFirestoreRepository($firebase.firestore, {
    collection: String(config.firestoreCollection || 'accounts'),
    document: String(config.firestoreDocument || 'current')
  })
}
