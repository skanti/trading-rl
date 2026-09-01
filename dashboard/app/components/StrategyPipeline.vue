<script setup lang="ts">
import {
  buildNaiveSchedule,
  formatCountdown,
  formatScheduleTime,
  formatZonedNow,
  type ScheduleConfig,
  type ScheduleEvent
} from '~/utils/schedule'
import type { MarketClock, StrategyState, TradingSchedule } from '~/types/dashboard'

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

const events = computed(() => buildNaiveSchedule(now.value, props.strategy, schedule.value))
const nextEvent = computed(() => events.value[0] ?? null)

const marketTarget = computed(() => {
  const value = props.market.is_open ? props.market.next_close : props.market.next_open
  if (!value) return null
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? null : parsed
})

const marketText = computed(() => {
  const state = props.market.is_open ? 'Market open' : 'Market closed'
  const target = marketTarget.value
  if (!target) return state
  const action = props.market.is_open ? 'closes' : 'opens'
  return `${state} · ${action} at ${formatScheduleTime(target, now.value, schedule.value.timeZone)}`
})

function eventText(event: ScheduleEvent): string {
  return `${event.label} · ${formatScheduleTime(event.at, now.value, schedule.value.timeZone)}`
}

function eventColor(event: ScheduleEvent): 'neutral' | 'info' | 'primary' | 'warning' {
  switch (event.key) {
    case 'exit': return 'warning'
    case 'rank': return 'info'
    case 'entry': return 'primary'
    default: return 'neutral'
  }
}

function countdownColor(event: ScheduleEvent): 'error' | 'info' | 'primary' | 'warning' | 'neutral' {
  return event.at < now.value ? 'error' : eventColor(event)
}
</script>

<template>
  <div class="min-w-0 rounded-lg border border-slate-800 bg-slate-900/40 px-2.5 py-2">
    <div class="flex min-w-0 items-center justify-between gap-3">
      <StatusBadge
        :status="strategy.status"
        appearance="eyebrow"
      />
      <span class="numeric shrink-0 text-[0.6875rem] text-slate-500">
        NY · {{ formatZonedNow(now, schedule.timeZone) }}
      </span>
    </div>

    <div
      v-if="nextEvent"
      class="mt-1.5 flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1"
    >
      <p class="min-w-0 text-sm font-semibold text-slate-100">
        {{ eventText(nextEvent) }}
      </p>
      <UBadge
        :color="countdownColor(nextEvent)"
        variant="subtle"
        size="sm"
        class="numeric shrink-0"
      >
        {{ formatCountdown(nextEvent.at, now) }}
      </UBadge>
    </div>

    <p class="mt-1 text-xs text-slate-500">
      {{ marketText }}
    </p>
  </div>
</template>
