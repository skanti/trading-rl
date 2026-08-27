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
  <div class="rounded-xl border border-slate-800 bg-slate-900/50 p-3 sm:p-4">
    <div class="flex items-center gap-1.5 text-[0.6875rem] font-medium uppercase tracking-wide text-slate-400 sm:gap-2 sm:text-xs">
      <UIcon
        v-if="icon"
        :name="icon"
        class="size-3.5"
      />
      {{ label }}
    </div>
    <p
      class="numeric mt-1.5 text-xl font-semibold sm:mt-2 sm:text-2xl"
      :class="tone"
    >
      {{ value }}
    </p>
    <p
      v-if="hint"
      class="numeric mt-0.5 text-xs text-slate-500 sm:mt-1"
    >
      {{ hint }}
    </p>
  </div>
</template>
