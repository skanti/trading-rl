<script setup lang="ts">
import {
  buildSessionTimeline,
  executionMilestone,
  formatCountdown,
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
  exitTime: props.schedule?.exit_time || '06:00'
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

const milestoneUi = {
  success: {
    indicator: 'bg-success text-inverted',
    separator: 'bg-success',
    description: 'numeric ms-auto shrink-0 whitespace-nowrap text-success text-xs/4 font-medium lg:ms-0 lg:mt-0.5'
  },
  warning: {
    indicator: 'bg-warning text-inverted',
    separator: 'bg-warning',
    description: 'numeric ms-auto shrink-0 whitespace-nowrap text-warning text-xs/4 font-medium lg:ms-0 lg:mt-0.5'
  },
  info: {
    indicator: 'bg-info text-inverted',
    separator: 'bg-info',
    description: 'numeric ms-auto shrink-0 whitespace-nowrap text-info text-xs/4 font-medium lg:ms-0 lg:mt-0.5'
  }
} as const

const timelineItems = computed<TimelineItem[]>(() => {
  const events = timeline.value?.events ?? []
  const execution = events.map(event => executionMilestone(event, props.strategy, now.value, schedule.value.timeZone))
  const activeExecutionIndex = execution.findIndex(state => state?.active)
  const nextIndex = activeExecutionIndex >= 0
    ? activeExecutionIndex
    : events.findIndex(event => event.at > now.value)

  return events.map((event, index) => {
    const executionState = execution[index]
    const completed = executionState?.completed ?? event.at <= now.value
    const warning = executionState?.warning ?? false
    const next = index === nextIndex
    const showCountdown = next || (event.key === 'next_open' && event.at > now.value)
    const description = executionState?.description
      ?? (showCountdown ? formatCountdown(event.at, now.value) : undefined)
    const tone = warning ? 'warning' : completed ? 'success' : showCountdown ? 'info' : 'neutral'
    return {
      value: index + 1,
      title: event.label,
      date: formatScheduleDateTime(event.at, now.value, schedule.value.timeZone),
      description,
      icon: completed ? warning ? 'i-lucide-triangle-alert' : 'i-lucide-check' : eventIcon(event),
      ui: tone === 'neutral'
        ? undefined
        : {
            ...milestoneUi[tone],
            separator: completed ? milestoneUi[tone].separator : undefined
          }
    }
  })
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
    class="min-w-0"
  >
    <template #header>
      <div class="flex min-w-0 items-start justify-between gap-3">
        <h2 class="text-sm font-semibold text-highlighted">
          Trading schedule
        </h2>
        <span class="numeric shrink-0 text-xs text-muted">
          NY · {{ formatZonedNow(now, schedule.timeZone) }}
        </span>
      </div>
    </template>

    <template v-if="timeline">
      <UTimeline
        :items="timelineItems"
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
    <p
      v-else
      class="text-sm text-muted"
    >
      Trading schedule unavailable.
    </p>
  </UCard>
</template>
