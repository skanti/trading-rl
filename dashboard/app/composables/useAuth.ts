import {
  browserLocalPersistence,
  onAuthStateChanged,
  setPersistence,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  type User
} from 'firebase/auth'
import type { FirebaseServices } from '~/plugins/firebase.client'

/**
 * Dashboard authentication.
 *
 * The login form asks for a bare username; Firebase Auth wants an email, so the two are
 * bridged with `auth.email_domain` from config.yaml. That keeps the credential Firebase
 * checks and the credential written in config.yaml as one thing -- there is no password
 * comparison in the client, which is why the password is never shipped in the bundle.
 *
 * When Firebase is not configured the app runs in demo mode and treats the visitor as
 * signed in, so a fresh checkout is usable before the project is provisioned.
 */
export function useAuth() {
  const { $firebase } = useNuxtApp() as unknown as { $firebase: FirebaseServices | null }
  const config = useRuntimeConfig().public

  const user = useState<User | null>('dashboard:user', () => null)
  const ready = useState<boolean>('dashboard:auth-ready', () => false)
  const demoMode = !$firebase

  if (import.meta.client && !ready.value) {
    if (demoMode) {
      ready.value = true
    } else {
      onAuthStateChanged($firebase!.auth, (current) => {
        user.value = current
        ready.value = true
      })
    }
  }

  const signedIn = computed(() => demoMode || Boolean(user.value))

  function emailFor(username: string): string {
    const trimmed = username.trim()
    if (trimmed.includes('@')) return trimmed
    return `${trimmed}@${String(config.authEmailDomain || '')}`
  }

  async function signIn(username: string, password: string): Promise<void> {
    if (demoMode) return
    await setPersistence($firebase!.auth, browserLocalPersistence)
    const credential = await signInWithEmailAndPassword(
      $firebase!.auth,
      emailFor(username),
      password
    )
    user.value = credential.user
  }

  async function signOut(): Promise<void> {
    if (demoMode) return
    await firebaseSignOut($firebase!.auth)
    user.value = null
  }

  /** Resolves once Firebase has reported the restored session, so guards do not flicker. */
  async function whenReady(): Promise<void> {
    if (ready.value) return
    await new Promise<void>((resolve) => {
      const stop = watch(ready, (value) => {
        if (value) {
          stop()
          resolve()
        }
      })
    })
  }

  return { user, ready, signedIn, demoMode, signIn, signOut, whenReady, emailFor }
}

/** Human-readable message for the Firebase Auth error codes the login form can hit. */
export function describeAuthError(error: unknown): string {
  const code = (error as { code?: string })?.code ?? ''
  switch (code) {
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found':
      return 'Incorrect username or password.'
    case 'auth/too-many-requests':
      return 'Too many attempts. Wait a moment and try again.'
    case 'auth/network-request-failed':
      return 'Network error reaching Firebase.'
    case 'auth/invalid-email':
      return 'That username is not valid.'
    case 'auth/operation-not-allowed':
    case 'auth/configuration-not-found':
      // Firebase reports this before any credential is checked, when the project has
      // no Email/Password provider enabled yet.
      return 'Email/password sign-in is not enabled on this Firebase project yet.'
    default:
      return (error as Error)?.message || 'Could not sign in.'
  }
}
