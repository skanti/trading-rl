<script setup lang="ts">
const props = defineProps<{ status?: string | null }>()

// The strategy's own vocabulary from live_overnight_liquidity's state machine.
const presentation = computed(() => {
  switch (props.status) {
    case 'open':
      return { color: 'primary' as const, label: 'Holding overnight' }
    case 'entering':
      return { color: 'info' as const, label: 'Entering' }
    case 'exiting':
      return { color: 'info' as const, label: 'Exiting' }
    case 'exit_queued':
      return { color: 'warning' as const, label: 'Exit queued' }
    case 'closed':
      return { color: 'neutral' as const, label: 'Flat' }
    case 'exit_incomplete':
      return { color: 'error' as const, label: 'Exit incomplete' }
    case 'entry_failed':
      return { color: 'error' as const, label: 'Entry failed' }
    default:
      return { color: 'neutral' as const, label: 'No position' }
  }
})
</script>

<template>
  <UBadge
    :color="presentation.color"
    variant="subtle"
    size="sm"
  >
    {{ presentation.label }}
  </UBadge>
</template>
