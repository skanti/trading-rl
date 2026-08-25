/**
 * Route guard. Waits for Firebase to restore any persisted session before deciding,
 * so a reload on a deep link does not bounce the visitor to the login page.
 */
export default defineNuxtRouteMiddleware(async (to) => {
  if (import.meta.server) return

  const { signedIn, whenReady } = useAuth()
  await whenReady()

  if (to.path === '/login') {
    return signedIn.value ? navigateTo('/') : undefined
  }

  if (!signedIn.value) {
    return navigateTo({ path: '/login', query: to.path === '/' ? {} : { redirect: to.fullPath } })
  }
})
