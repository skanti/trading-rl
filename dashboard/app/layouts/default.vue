<script setup lang="ts">
import { formatRelative } from '~/utils/format'

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
let timer: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  timer = setInterval(() => {
    now.value = new Date()
  }, 30_000)
})
onBeforeUnmount(() => clearInterval(timer))

const updated = computed(() => formatRelative(snapshot.value?.updated_at, now.value))

async function handleSignOut() {
  await signOut()
  await navigateTo('/login')
}
</script>

<template>
  <div class="min-h-screen min-w-0 overflow-x-clip bg-slate-950 text-slate-100">
    <header class="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
      <div class="mx-auto grid w-full max-w-6xl grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2 px-2 py-3 sm:px-4 md:flex md:gap-x-6">
        <NuxtLink
          to="/"
          class="flex min-w-0 items-center gap-2"
        >
          <span class="flex size-7 items-center justify-center rounded-md bg-emerald-500/15">
            <UIcon
              name="i-lucide-trending-up"
              class="size-4 text-emerald-400"
            />
          </span>
          <span class="truncate text-sm font-semibold">{{ config.dashboardTitle }}</span>
        </NuxtLink>

        <nav class="col-span-2 row-start-2 flex min-w-0 items-center gap-1 md:col-auto md:row-auto">
          <NuxtLink
            v-for="link in links"
            :key="link.to"
            :to="link.to"
            class="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition"
            :class="route.path === link.to
              ? 'bg-slate-800 text-white'
              : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'"
          >
            <UIcon
              :name="link.icon"
              class="size-4"
            />
            {{ link.label }}
          </NuxtLink>
        </nav>

        <div class="col-start-2 row-start-1 flex items-center gap-1 sm:gap-2 md:col-auto md:row-auto md:ml-auto md:gap-3">
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
          <UButton
            icon="i-lucide-log-out"
            color="neutral"
            variant="ghost"
            size="sm"
            aria-label="Sign out"
            @click="handleSignOut"
          />
        </div>
      </div>
    </header>

    <main class="mx-auto min-w-0 w-full max-w-6xl px-2 py-5 sm:px-4 sm:py-6">
      <slot />
    </main>
  </div>
</template>
