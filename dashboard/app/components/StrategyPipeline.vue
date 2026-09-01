<script setup lang="ts">
import {
  buildSessionTimeline,
  formatCountdown,
  lastCompletedTimelineIndex,
  formatScheduleDateTime,
  formatZonedNow,
  type ScheduleConfig,
  type ScheduleEvent
} from '~/utils/schedule'
import type { MarketClock, StrategyState, TradingSchedule } from '~/types/dashboard'
import type { TimelineItem } from '@nuxt/ui'

const props = defineProps<{
  strategy: StrategyState
  market: MarketClock
  schedule?: TradingSchedule
}>()

const schedule = computed<ScheduleConfig>(() => ({
  timeZone: props.schedule?.time_zone || 'America/New_York',
  rankingTime: props.schedule?.ranking_time || '14:00',
  entryTime: props.schedule?.entry_time || '15:45',
  exitTime: props.schedule?.exit_time || '08:00'
}))

const now = ref(new Date())
let timer: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  timer = setInterval(() => {
    now.value = new Date()
  }, 30_000)
})
onBeforeUnmount(() => clearInterval(timer))

const timeline = computed(() => buildSessionTimeline(
  now.value,
  props.strategy,
  schedule.value,
  props.market
))
const timelineItems = computed<TimelineItem[]>(() => {
  const events = timeline.value?.events ?? []
  const nextIndex = events.findIndex(event => event.at > now.value)
  return events.map((event, index) => {
    const completed = event.at <= now.value
    const next = index === nextIndex
    return {
      value: index + 1,
      title: event.label,
      date: formatScheduleDateTime(event.at, now.value, schedule.value.timeZone),
      description: next ? formatCountdown(event.at, now.value) : undefined,
      icon: completed ? 'i-lucide-check' : eventIcon(event),
      ui: next
        ? {
            indicator: 'bg-info text-inverted',
            description: 'numeric ms-auto shrink-0 whitespace-nowrap text-info text-xs/4 font-medium lg:ms-0 lg:mt-0.5'
          }
        : undefined
    }
  })
})
const timelineStep = computed(() => {
  return lastCompletedTimelineIndex(timeline.value?.events ?? [], now.value)
})

function eventIcon(event: ScheduleEvent): string {
  switch (event.key) {
    case 'market_open': return 'i-lucide-sunrise'
    case 'rank': return 'i-lucide-list-ordered'
    case 'entry': return 'i-lucide-log-in'
    case 'market_close': return 'i-lucide-sunset'
    case 'exit': return 'i-lucide-log-out'
    case 'next_open': return 'i-lucide-bell-ring'
  }
}
</script>

<template>
  <UCard
    variant="subtle"
    class="min-w-0"
  >
    <template #header>
      <div class="flex min-w-0 items-start justify-between gap-3">
        <div class="min-w-0">
          <h2 class="text-sm font-semibold text-highlighted">
            Trading schedule
          </h2>
        </div>
        <span class="numeric shrink-0 text-xs text-muted">
          NY · {{ formatZonedNow(now, schedule.timeZone) }}
        </span>
      </div>
    </template>

    <template v-if="timeline">
      <UTimeline
        :items="timelineItems"
        :model-value="timelineStep"
        size="xs"
        class="lg:hidden"
        :ui="{
          root: 'gap-0',
          item: 'gap-2',
          container: 'gap-1',
          wrapper: 'mt-0 flex items-baseline gap-2 pb-2',
          date: 'shrink-0 text-xs/4',
          title: 'shrink-0 text-xs/4'
        }"
      />
      <UTimeline
        :items="timelineItems"
        :model-value="timelineStep"
        orientation="horizontal"
        size="xs"
        class="hidden lg:flex"
        :ui="{
          root: 'gap-1',
          item: 'gap-2',
          container: 'gap-1',
          wrapper: 'pe-2',
          date: 'text-xs/4',
          title: 'text-xs/4'
        }"
      />
    </template>
  </UCard>
</template>
