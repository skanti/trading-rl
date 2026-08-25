<script setup lang="ts">
withDefaults(defineProps<{
  label: string
  value: string
  hint?: string
  tone?: string
  icon?: string
}>(), {
  hint: undefined,
  tone: 'text-white',
  icon: undefined
})
</script>

<template>
  <div class="rounded-xl border border-slate-800 bg-slate-900/50 p-4">
    <div class="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-400">
      <UIcon
        v-if="icon"
        :name="icon"
        class="size-3.5"
      />
      {{ label }}
    </div>
    <p
      class="numeric mt-2 text-2xl font-semibold"
      :class="tone"
    >
      {{ value }}
    </p>
    <p
      v-if="hint"
      class="numeric mt-1 text-xs text-slate-500"
    >
      {{ hint }}
    </p>
  </div>
</template>
