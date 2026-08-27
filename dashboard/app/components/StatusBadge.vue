<script setup lang="ts">
const props = withDefaults(defineProps<{
  status?: string | null
  appearance?: 'badge' | 'eyebrow'
}>(), {
  appearance: 'badge'
})

// The strategy's own vocabulary from live_overnight_liquidity's state machine.
const presentation = computed(() => {
  switch (props.status) {
    case 'ranking':
      return { color: 'info' as const, dot: 'bg-sky-400', text: 'text-sky-300', label: 'Ranking stocks' }
    case 'planned':
      return { color: 'neutral' as const, dot: 'bg-slate-500', text: 'text-slate-300', label: 'Awaiting entry' }
    case 'open':
      return { color: 'neutral' as const, dot: 'bg-slate-500', text: 'text-slate-300', label: 'Position open' }
    case 'entering':
      return { color: 'info' as const, dot: 'bg-sky-400', text: 'text-sky-300', label: 'Opening position' }
    case 'exit_plan':
      return { color: 'neutral' as const, dot: 'bg-slate-500', text: 'text-slate-300', label: 'Awaiting exit' }
    case 'exiting':
      return { color: 'info' as const, dot: 'bg-sky-400', text: 'text-sky-300', label: 'Closing position' }
    case 'exit_queued':
      return { color: 'warning' as const, dot: 'bg-amber-400', text: 'text-amber-300', label: 'Exit queued' }
    case 'closed':
      return { color: 'neutral' as const, dot: 'bg-slate-500', text: 'text-slate-400', label: 'Flat' }
    case 'exit_incomplete':
      return { color: 'error' as const, dot: 'bg-red-400', text: 'text-red-300', label: 'Exit incomplete' }
    case 'entry_failed':
      return { color: 'error' as const, dot: 'bg-red-400', text: 'text-red-300', label: 'Entry failed' }
    default:
      return { color: 'neutral' as const, dot: 'bg-slate-500', text: 'text-slate-400', label: 'No position' }
  }
})
</script>

<template>
  <span
    v-if="appearance === 'eyebrow'"
    class="inline-flex min-w-0 items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide"
    :class="presentation.text"
  >
    <span
      class="size-1.5 shrink-0 rounded-full"
      :class="presentation.dot"
    />
    {{ presentation.label }}
  </span>
  <UBadge
    v-else
    :color="presentation.color"
    variant="subtle"
    size="sm"
  >
    {{ presentation.label }}
  </UBadge>
</template>
