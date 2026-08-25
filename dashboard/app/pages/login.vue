<script setup lang="ts">
import { describeAuthError } from '~/composables/useAuth'

definePageMeta({ layout: false })

const { signIn, demoMode } = useAuth()
const route = useRoute()
const config = useRuntimeConfig().public

const username = ref('')
const password = ref('')
const error = ref<string | null>(null)
const busy = ref(false)

async function submit() {
  if (busy.value) return
  busy.value = true
  error.value = null
  try {
    await signIn(username.value, password.value)
    await navigateTo(String(route.query.redirect ?? '/'))
  } catch (cause) {
    error.value = describeAuthError(cause)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-slate-100">
    <div class="w-full max-w-sm">
      <div class="mb-6 flex items-center gap-2.5">
        <span class="flex size-9 items-center justify-center rounded-lg bg-emerald-500/15">
          <UIcon
            name="i-lucide-trending-up"
            class="size-5 text-emerald-400"
          />
        </span>
        <div>
          <h1 class="text-base font-semibold">
            {{ config.dashboardTitle }}
          </h1>
          <p class="text-xs text-slate-500">
            Sign in to view the account
          </p>
        </div>
      </div>

      <form
        class="space-y-4 rounded-xl border border-slate-800 bg-slate-900/50 p-5"
        @submit.prevent="submit"
      >
        <div class="space-y-1.5">
          <label
            for="username"
            class="text-xs font-medium text-slate-400"
          >Username</label>
          <UInput
            id="username"
            v-model="username"
            autocomplete="username"
            autocapitalize="none"
            spellcheck="false"
            placeholder="username"
            size="lg"
            class="w-full"
          />
        </div>

        <div class="space-y-1.5">
          <label
            for="password"
            class="text-xs font-medium text-slate-400"
          >Password</label>
          <UInput
            id="password"
            v-model="password"
            type="password"
            autocomplete="current-password"
            placeholder="••••••••"
            size="lg"
            class="w-full"
          />
        </div>

        <p
          v-if="error"
          class="flex items-start gap-1.5 text-sm text-rose-400"
        >
          <UIcon
            name="i-lucide-triangle-alert"
            class="mt-0.5 size-4 shrink-0"
          />
          {{ error }}
        </p>

        <UButton
          type="submit"
          block
          size="lg"
          :loading="busy"
          :disabled="!username || !password"
        >
          Sign in
        </UButton>

        <p
          v-if="demoMode"
          class="text-xs text-amber-400/80"
        >
          Firebase is not configured, so the dashboard is serving demo data. Any
          credentials will get you in.
        </p>
      </form>
    </div>
  </div>
</template>
