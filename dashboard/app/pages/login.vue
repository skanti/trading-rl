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

      <UCard>
        <form
          class="space-y-4"
          @submit.prevent="submit"
        >
          <UFormField
            label="Username"
            name="username"
          >
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
          </UFormField>

          <UFormField
            label="Password"
            name="password"
          >
            <UInput
              id="password"
              v-model="password"
              type="password"
              autocomplete="current-password"
              placeholder="••••••••"
              size="lg"
              class="w-full"
            />
          </UFormField>

          <UAlert
            v-if="error"
            color="error"
            variant="subtle"
            icon="i-lucide-triangle-alert"
            title="Could not sign in"
            :description="error"
          />

          <UButton
            type="submit"
            block
            size="lg"
            :loading="busy"
            :disabled="!username || !password"
          >
            Sign in
          </UButton>

          <UAlert
            v-if="demoMode"
            color="warning"
            variant="subtle"
            title="Demo data"
            description="Firebase is not configured. Any credentials will sign in."
          />
        </form>
      </UCard>
    </div>
  </div>
</template>
