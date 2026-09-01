<script setup lang="ts">
import {
  buildSessionTimeline,
  formatCountdown,
  formatScheduleDateTime,
  formatScheduleTime,
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

interface ExecutionMilestone {
  active: boolean
  completed: boolean
  warning: boolean
  description?: string
}

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

function completionTime(value: string | null, scheduledAt: Date): Date | null {
  if (!value) return null
  const completedAt = new Date(value)
  if (Number.isNaN(completedAt.getTime()) || completedAt < scheduledAt) return null
  return completedAt
}

function executionMilestone(event: ScheduleEvent): ExecutionMilestone | null {
  if (event.key === 'entry') {
    const completedAt = completionTime(props.strategy.entry_completed_at, event.at)
    const total = props.strategy.symbols.length
    const filled = props.strategy.filled_symbols.length
    if (completedAt) {
      const warning = total > 0 && filled < total
      return {
        active: false,
        completed: true,
        warning,
        description: warning
          ? `${filled}/${total} filled · ${formatScheduleTime(completedAt, now.value, schedule.value.timeZone)}`
          : `Filled · ${formatScheduleTime(completedAt, now.value, schedule.value.timeZone)}`
      }
    }
    if (event.at <= now.value) {
      return {
        active: true,
        completed: false,
        warning: false,
        description: total > 0 ? `Filling · ${filled}/${total}` : 'Filling'
      }
    }
  }

  if (event.key === 'exit') {
    const completedAt = completionTime(props.strategy.exit_completed_at, event.at)
    const remaining = props.strategy.remaining_symbols.length
    if (completedAt) {
      return {
        active: false,
        completed: true,
        warning: remaining > 0,
        description: remaining > 0
          ? `${remaining} remaining · ${formatScheduleTime(completedAt, now.value, schedule.value.timeZone)}`
          : `Exited · ${formatScheduleTime(completedAt, now.value, schedule.value.timeZone)}`
      }
    }
    if (event.at <= now.value) {
      return {
        active: true,
        completed: false,
        warning: false,
        description: remaining > 0 ? `Exiting · ${remaining} remaining` : 'Exiting'
      }
    }
  }

  return null
}

const timelineItems = computed<TimelineItem[]>(() => {
  const events = timeline.value?.events ?? []
  const execution = events.map(event => executionMilestone(event))
  const activeExecutionIndex = execution.findIndex(state => state?.active)
  const nextIndex = activeExecutionIndex >= 0
    ? activeExecutionIndex
    : events.findIndex(event => event.at > now.value)

  return events.map((event, index) => {
    const executionState = execution[index]
    const completed = executionState?.completed ?? event.at <= now.value
    const warning = executionState?.warning ?? false
    const next = index === nextIndex
    const description = executionState?.description
      ?? (next ? formatCountdown(event.at, now.value) : undefined)
    const tone = warning ? 'warning' : completed ? 'success' : next ? 'info' : 'neutral'
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
  </UCard>
</template>
