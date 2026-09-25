<script setup lang="ts">
import { formatRelative } from '~/utils/format'
import type { DropdownMenuItem } from '@nuxt/ui'

const route = useRoute()
const { signOut, demoMode } = useAuth()
const { snapshot, pending, refresh } = useSnapshot()
const config = useRuntimeConfig().public

const links = [
  { label: 'Overview', to: '/', icon: 'i-lucide-layout-dashboard' },
  { label: 'Positions', to: '/positions', icon: 'i-lucide-layers' },
  { label: 'History', to: '/history', icon: 'i-lucide-history' }
]

// Recomputed on a ticker so "3 minutes ago" does not freeze on a long-open tab.
const now = ref(new Date())
let clockTimer: ReturnType<typeof setInterval> | undefined
let refreshTimer: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  clockTimer = setInterval(() => {
    now.value = new Date()
  }, 30_000)
  refreshTimer = setInterval(() => {
    // Avoid background Firestore reads while the installed app or tab is hidden.
    if (document.visibilityState === 'visible') void refresh()
  }, 120_000)
})
onBeforeUnmount(() => {
  clearInterval(clockTimer)
  clearInterval(refreshTimer)
})

const updated = computed(() => formatRelative(snapshot.value?.updated_at, now.value))

async function handleSignOut() {
  await signOut()
  await navigateTo('/login')
}

const dashboardMenu = computed<DropdownMenuItem[][]>(() => [links.map(link => ({
  ...link,
  active: route.path === link.to
})), [
  {
    label: 'Sign out',
    icon: 'i-lucide-log-out',
    color: 'error',
    onSelect: () => void handleSignOut()
  }
]])
</script>

<template>
  <div class="min-h-screen min-w-0 overflow-x-clip bg-slate-950 text-slate-100">
    <header class="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
      <div class="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 p-2 md:gap-6">
        <div class="flex min-w-0 items-center gap-1">
          <UDropdownMenu
            :items="dashboardMenu"
            :content="{ align: 'start' }"
          >
            <UButton
              color="neutral"
              variant="ghost"
              size="sm"
              aria-label="Open dashboard menu"
              class="-ml-1 px-1"
            >
              <span class="flex size-7 items-center justify-center rounded-md bg-emerald-500/15">
                <UIcon
                  name="i-lucide-trending-up"
                  class="size-4 text-emerald-400"
                />
              </span>
              <UIcon
                name="i-lucide-chevron-down"
                class="size-3 text-muted"
              />
            </UButton>
          </UDropdownMenu>

          <NuxtLink
            to="/"
            class="truncate text-sm font-semibold"
          >
            {{ config.dashboardTitle }}
          </NuxtLink>
        </div>

        <div class="flex shrink-0 items-center gap-1 sm:gap-2 md:gap-3">
          <UBadge
            v-if="demoMode"
            class="hidden sm:inline-flex"
            color="warning"
            variant="subtle"
            size="sm"
          >
            Demo data
          </UBadge>
          <span class="hidden text-xs text-slate-500 sm:inline">
            Updated {{ updated }}
          </span>
          <UButton
            icon="i-lucide-refresh-cw"
            color="neutral"
            variant="ghost"
            size="sm"
            :loading="pending"
            aria-label="Refresh"
            @click="refresh()"
          />
        </div>
      </div>
    </header>

    <main class="mx-auto min-w-0 w-full max-w-6xl px-2 py-5 sm:px-4 sm:py-6">
      <slot />
    </main>
  </div>
</template>
