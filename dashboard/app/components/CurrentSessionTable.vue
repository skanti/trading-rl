<script setup lang="ts">
import {
  formatCurrency,
  formatDay,
  formatPercent,
  formatSignedCurrency,
  formatSignedPercent,
  toneClass
} from '~/utils/format'
import type { Position, StrategyState } from '~/types/dashboard'
import type { TableColumn } from '@nuxt/ui'

const props = defineProps<{
  strategy: StrategyState
  positions: Position[]
}>()

const summary = computed(() => {
  const deployed = props.positions.reduce((sum, position) => sum + (position.cost_basis ?? 0), 0)
  const marketValue = props.positions.reduce((sum, position) => sum + (position.market_value ?? 0), 0)
  const openPnl = props.positions.reduce((sum, position) => sum + (position.unrealized_pl ?? 0), 0)
  const winners = props.positions.filter(position => (position.unrealized_pl ?? 0) > 0).length
  const count = props.positions.length
  return {
    deployed,
    marketValue,
    openPnl,
    openReturn: deployed > 0 ? openPnl / deployed : 0,
    winners,
    count,
    winRate: count > 0 ? winners / count : null,
    utilization: props.strategy.budget > 0 ? deployed / props.strategy.budget : null
  }
})

interface SessionMetricRow {
  metric: string
  value: string
  detail: string
  tone?: string
}

const columns: TableColumn<SessionMetricRow>[] = [
  {
    accessorKey: 'metric',
    header: 'Metric',
    meta: { class: { td: 'whitespace-normal' } }
  },
  {
    accessorKey: 'value',
    header: 'Value',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'detail',
    header: 'Detail',
    meta: {
      class: {
        th: 'hidden text-right sm:table-cell',
        td: 'hidden text-right sm:table-cell'
      }
    }
  }
]

const rows = computed<SessionMetricRow[]>(() => [
  {
    metric: 'Deployed capital',
    value: formatCurrency(summary.value.deployed),
    detail: summary.value.utilization === null
      ? '—'
      : `${formatPercent(summary.value.utilization, 1)} of budget`
  },
  {
    metric: 'Open P&L',
    value: formatSignedCurrency(summary.value.openPnl),
    detail: formatSignedPercent(summary.value.openReturn),
    tone: toneClass(summary.value.openPnl)
  },
  {
    metric: 'Per-symbol win rate',
    value: summary.value.winRate === null ? '—' : formatPercent(summary.value.winRate, 1),
    detail: summary.value.count
      ? `${summary.value.winners} / ${summary.value.count} profitable`
      : 'No open symbols'
  },
  {
    metric: 'Position value',
    value: formatCurrency(summary.value.marketValue),
    detail: `${summary.value.count} position${summary.value.count === 1 ? '' : 's'}`
  },
  {
    metric: 'Holding window',
    value: formatDay(props.strategy.entry_date),
    detail: `to ${formatDay(props.strategy.exit_date)}`
  }
])
</script>

<template>
  <UCard
    variant="subtle"
    class="min-w-0"
  >
    <template #header>
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p class="text-xs uppercase tracking-wide text-muted">
            Current trading session
          </p>
          <h1 class="mt-0.5 text-base font-semibold text-highlighted">
            {{ strategy.entry_date ? formatDay(strategy.entry_date) : 'Awaiting the next entry' }}
          </h1>
        </div>
        <StatusBadge :status="strategy.status" />
      </div>
    </template>

    <UTable
      :data="rows"
      :columns="columns"
      class="min-w-0"
    >
      <template #value-cell="{ row }">
        <div class="flex flex-col items-end">
          <span
            class="numeric font-semibold"
            :class="row.original.tone ?? 'text-highlighted'"
          >
            {{ row.original.value }}
          </span>
          <span
            class="numeric mt-0.5 whitespace-normal text-xs text-muted sm:hidden"
            :class="row.original.tone"
          >
            {{ row.original.detail }}
          </span>
        </div>
      </template>
      <template #detail-cell="{ row }">
        <span
          class="numeric text-muted"
          :class="row.original.tone"
        >
          {{ row.original.detail }}
        </span>
      </template>
    </UTable>
  </UCard>
</template>
