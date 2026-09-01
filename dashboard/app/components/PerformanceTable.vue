<script setup lang="ts">
import { parseDate, type DateValue } from '@internationalized/date'
import {
  formatCurrency,
  formatDay,
  formatPercent,
  formatSignedCurrency,
  formatSignedPercent,
  profitFactorToneClass,
  toneClass
} from '~/utils/format'
import {
  performanceWindow,
  periodMetrics,
  type PerformancePeriod
} from '~/utils/performance'
import type { EquityPoint } from '~/types/dashboard'
import type { TableColumn } from '@nuxt/ui'

const props = defineProps<{
  points: EquityPoint[]
  asOf: string
}>()

type ShortcutPeriod = Exclude<PerformancePeriod, 'custom'>

interface SelectedDateRange {
  start: DateValue | undefined
  end: DateValue | undefined
}

const shortcuts: Array<{ label: string, title: string, value: ShortcutPeriod }> = [
  { label: 'WTD', title: 'Week to date', value: 'this_week' },
  { label: 'MTD', title: 'Month to date', value: 'this_month' },
  { label: 'YTD', title: 'This year', value: 'this_year' },
  { label: 'All', title: 'Since inception', value: 'inception' }
]

const selectedPeriod = ref<PerformancePeriod>('this_year')
const firstDay = computed(() => props.points.reduce(
  (earliest, point) => point.day < earliest ? point.day : earliest,
  props.asOf
))

function presetRange(period: ShortcutPeriod): SelectedDateRange {
  const preset = performanceWindow(period, props.asOf)
  return {
    start: parseDate(preset.start ?? firstDay.value),
    end: parseDate(preset.end)
  }
}

const selectedRange = shallowRef<SelectedDateRange>(presetRange('this_year'))
const calendarOpen = ref(false)

const window = computed(() => performanceWindow(
  'custom',
  props.asOf,
  selectedRange.value.start?.toString() ?? firstDay.value,
  selectedRange.value.end?.toString() ?? props.asOf
))
const metrics = computed(() => periodMetrics(props.points, window.value))
const minDate = computed(() => parseDate(firstDay.value))
const maxDate = computed(() => parseDate(props.asOf))

watch([() => props.asOf, firstDay], () => {
  if (selectedPeriod.value !== 'custom') selectShortcut(selectedPeriod.value)
})

function selectShortcut(period: ShortcutPeriod) {
  selectedPeriod.value = period
  selectedRange.value = presetRange(period)
}

function selectDateRange(value: unknown) {
  if (!value || Array.isArray(value) || typeof value !== 'object' || !('start' in value)) return
  const range = value as SelectedDateRange
  if (!range.start) return

  selectedPeriod.value = 'custom'
  selectedRange.value = range
  if (range.end) calendarOpen.value = false
}

function formatRangeDay(value: DateValue | undefined, includeYear: boolean): string {
  if (!value) return '…'
  const date = new Date(`${value.toString()}T00:00:00Z`)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: includeYear ? 'numeric' : undefined,
    timeZone: 'UTC'
  })
}

const rangeLabel = computed(() => {
  const start = selectedRange.value.start
  const end = selectedRange.value.end
  const differentYears = Boolean(start && end && start.year !== end.year)
  return `${formatRangeDay(start, differentYears)} – ${formatRangeDay(end, differentYears)}`
})

function formatRatio(value: number | null, digits = 2): string {
  if (value === null || Number.isNaN(value)) return '—'
  if (!Number.isFinite(value)) return '∞'
  return value.toFixed(digits)
}

interface PerformanceRow {
  label: string
  value: string
  detail: string
  tone: string
}

const columns: TableColumn<PerformanceRow>[] = [
  {
    accessorKey: 'label',
    header: 'KPI',
    meta: { class: { td: 'whitespace-normal' } }
  },
  {
    accessorKey: 'value',
    header: 'Value',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'detail',
    header: 'Context',
    meta: {
      class: {
        th: 'hidden text-right sm:table-cell',
        td: 'hidden text-right sm:table-cell'
      }
    }
  }
]

const rows = computed<PerformanceRow[]>(() => [
  {
    label: 'Net P&L',
    value: formatSignedCurrency(metrics.value.pnl),
    detail: formatSignedPercent(metrics.value.totalReturn),
    tone: toneClass(metrics.value.pnl)
  },
  {
    label: 'Sessions',
    value: metrics.value.sessions.toLocaleString(),
    detail: 'Confirmed baskets',
    tone: 'text-slate-200'
  },
  {
    label: 'Trades',
    value: metrics.value.trades.toLocaleString(),
    detail: 'Completed symbol round trips',
    tone: 'text-slate-200'
  },
  {
    label: 'Annualized return',
    value: formatSignedPercent(metrics.value.annualizedReturn),
    detail: `${metrics.value.sessions} closed session${metrics.value.sessions === 1 ? '' : 's'}`,
    tone: toneClass(metrics.value.annualizedReturn)
  },
  {
    label: 'Average session',
    value: formatSignedPercent(metrics.value.averageReturn),
    detail: 'Arithmetic mean',
    tone: toneClass(metrics.value.averageReturn)
  },
  {
    label: 'Sharpe ratio',
    value: formatRatio(metrics.value.sharpe),
    detail: 'Annualized · 0% cash rate',
    tone: toneClass(metrics.value.sharpe)
  },
  {
    label: 'Sortino ratio',
    value: formatRatio(metrics.value.sortino),
    detail: 'Downside deviation',
    tone: toneClass(metrics.value.sortino)
  },
  {
    label: 'Annualized volatility',
    value: formatPercent(metrics.value.annualizedVolatility),
    detail: 'Closed-session returns',
    tone: 'text-slate-200'
  },
  {
    label: 'Maximum drawdown',
    value: formatCurrency(metrics.value.maxDrawdown),
    detail: formatPercent(metrics.value.maxDrawdownPct),
    tone: metrics.value.maxDrawdown > 0 ? 'text-rose-400' : 'text-slate-400'
  },
  {
    label: 'Session win rate',
    value: formatPercent(metrics.value.winRate, 1),
    detail: metrics.value.sessions ? `${Math.round((metrics.value.winRate ?? 0) * metrics.value.sessions)} winners` : '—',
    tone: 'text-slate-200'
  },
  {
    label: 'Profit factor',
    value: formatRatio(metrics.value.profitFactor),
    detail: 'Gross gains ÷ gross losses',
    tone: profitFactorToneClass(metrics.value.profitFactor)
  },
  {
    label: 'Best session',
    value: formatSignedPercent(metrics.value.bestSession?.value),
    detail: formatDay(metrics.value.bestSession?.day),
    tone: toneClass(metrics.value.bestSession?.value)
  },
  {
    label: 'Worst session',
    value: formatSignedPercent(metrics.value.worstSession?.value),
    detail: formatDay(metrics.value.worstSession?.day),
    tone: toneClass(metrics.value.worstSession?.value)
  }
])
</script>

<template>
  <UCard
    class="min-w-0 max-w-full"
  >
    <template #header>
      <div>
        <div>
          <h2 class="text-sm font-semibold text-highlighted">
            Realized performance
          </h2>
          <p class="mt-0.5 text-xs text-muted">
            {{ formatDay(window.start ?? firstDay) }}–{{ formatDay(window.end) }} · confirmed baskets only
          </p>
        </div>

        <div class="mt-3 flex min-w-0 items-center justify-between gap-2">
          <div class="min-w-0 flex-1 overflow-x-auto">
            <UFieldGroup
              size="xs"
              class="w-max"
            >
              <UButton
                v-for="shortcut in shortcuts"
                :key="shortcut.value"
                color="neutral"
                :variant="selectedPeriod === shortcut.value ? 'solid' : 'outline'"
                :title="shortcut.title"
                :aria-pressed="selectedPeriod === shortcut.value"
                @click="selectShortcut(shortcut.value)"
              >
                {{ shortcut.label }}
              </UButton>
            </UFieldGroup>
          </div>

          <UPopover
            v-model:open="calendarOpen"
            :content="{ align: 'end' }"
          >
            <UButton
              color="neutral"
              variant="outline"
              icon="i-lucide-calendar-days"
              class="numeric shrink-0"
              aria-label="Select performance date range"
            >
              {{ rangeLabel }}
            </UButton>
            <template #content>
              <UCalendar
                :model-value="selectedRange"
                range
                :min-value="minDate"
                :max-value="maxDate"
                class="p-2"
                @update:model-value="selectDateRange"
              />
            </template>
          </UPopover>
        </div>
      </div>
    </template>

    <UEmpty
      v-if="!points.length"
      icon="i-lucide-chart-no-axes-combined"
      title="No realized performance yet"
      description="Performance appears after the first closed basket's fees are confirmed."
      class="py-8"
    />

    <template v-else>
      <UAlert
        v-if="!metrics.sessions"
        color="neutral"
        variant="subtle"
        icon="i-lucide-info"
        title="No confirmed baskets in this period"
        description="Risk metrics exclude sessions whose fees are still pending."
        class="rounded-none"
      />

      <UTable
        :data="rows"
        :columns="columns"
        class="min-w-0"
      >
        <template #value-cell="{ row }">
          <div class="flex flex-col items-end">
            <span
              class="numeric font-semibold"
              :class="row.original.tone"
            >
              {{ row.original.value }}
            </span>
            <span class="numeric mt-0.5 whitespace-normal text-xs text-muted sm:hidden">
              {{ row.original.detail }}
            </span>
          </div>
        </template>
        <template #detail-cell="{ row }">
          <span class="numeric text-muted">
            {{ row.original.detail }}
          </span>
        </template>
      </UTable>
    </template>
  </UCard>
</template>
