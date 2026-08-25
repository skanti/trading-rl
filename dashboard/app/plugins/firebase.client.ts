import { type FirebaseApp, getApps, initializeApp } from 'firebase/app'
import { type Auth, getAuth } from 'firebase/auth'
import { type Firestore, getFirestore } from 'firebase/firestore'

export interface FirebaseServices {
  app: FirebaseApp
  auth: Auth
  firestore: Firestore
}

/**
 * Initialise Firebase once per browser session.
 *
 * The values come from config.yaml via runtimeConfig. When they are absent -- a fresh
 * checkout before the project is provisioned -- `firebase` is provided as null and the
 * app falls back to the demo adapter rather than crashing on boot.
 */
export default defineNuxtPlugin(() => {
  const config = useRuntimeConfig().public

  const apiKey = String(config.firebaseApiKey || '')
  const projectId = String(config.firebaseProjectId || '')
  const appId = String(config.firebaseAppId || '')

  if (!apiKey || !projectId || !appId) {
    return { provide: { firebase: null as FirebaseServices | null } }
  }

  const app = getApps()[0] ?? initializeApp({
    apiKey,
    authDomain: String(config.firebaseAuthDomain || ''),
    projectId,
    storageBucket: String(config.firebaseStorageBucket || ''),
    messagingSenderId: String(config.firebaseMessagingSenderId || ''),
    appId
  })

  const services: FirebaseServices = {
    app,
    auth: getAuth(app),
    firestore: getFirestore(app)
  }

  return { provide: { firebase: services } }
})
