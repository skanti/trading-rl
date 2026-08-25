<script setup lang="ts">
const { sessions, ensureLoaded, pending, snapshot } = useSnapshot()

await ensureLoaded()
</script>

<template>
  <div class="space-y-5">
    <div>
      <h1 class="text-sm font-semibold text-white">
        History
      </h1>
      <p class="text-xs text-slate-500">
        Every trading day the daemon recorded, newest first. Select a row to see the
        per-symbol fills.
      </p>
    </div>

    <USkeleton
      v-if="pending && !snapshot"
      class="h-96 rounded-xl"
    />
    <SessionsTable
      v-else
      :sessions="sessions"
    />
  </div>
</template>
