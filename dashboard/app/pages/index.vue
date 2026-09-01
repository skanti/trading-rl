<script setup lang="ts">
import { formatCurrency } from '~/utils/format'
import type { TableColumn } from '@nuxt/ui'

const { snapshot, sessions, pending, error, ensureLoaded } = useSnapshot()

await ensureLoaded()

const account = computed(() => snapshot.value?.account ?? {})
const inception = computed(() => snapshot.value?.performance?.inception)
const curve = computed(() => snapshot.value?.equity_curve ?? [])
const openPerformance = computed(() => {
  const positions = snapshot.value?.positions ?? []
  const pnl = positions.reduce((sum, position) => sum + (position.unrealized_pl ?? 0), 0)
  return { pnl, positions: positions.length }
})
const recentSessions = computed(() => sessions.value.slice(0, 8))

interface AccountRow {
  metric: string
  value: string
  emphasis?: boolean
  muted?: boolean
}

const accountColumns: TableColumn<AccountRow>[] = [
  { accessorKey: 'metric', header: 'Metric' },
  {
    accessorKey: 'value',
    header: 'Value',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  }
]

const accountRows = computed<AccountRow[]>(() => [
  { metric: 'Equity', value: formatCurrency(account.value.equity), emphasis: true },
  { metric: 'Cash', value: formatCurrency(account.value.cash) },
  { metric: 'Long market value', value: formatCurrency(account.value.long_market_value) },
  { metric: 'Buying power', value: formatCurrency(account.value.buying_power) },
  { metric: 'Account', value: account.value.account_number ?? '—', muted: true }
])
</script>

<template>
  <div class="space-y-5">
    <UAlert
      v-if="error"
      color="error"
      variant="subtle"
      icon="i-lucide-triangle-alert"
      title="Could not load the snapshot"
      :description="error"
    />

    <div
      v-if="pending && !snapshot"
      class="space-y-5"
    >
      <USkeleton class="h-64 rounded-xl" />
      <USkeleton class="h-52 rounded-xl" />
      <USkeleton class="h-80 rounded-xl" />
    </div>

    <UAlert
      v-else-if="!snapshot"
      color="warning"
      variant="subtle"
      icon="i-lucide-database"
      title="Nothing published yet"
      description="Run `python scripts/dashboard_daemon.py --once` to publish the first account snapshot."
    />

    <template v-else>
      <CurrentSessionTable
        :strategy="snapshot.strategy"
        :positions="snapshot.positions ?? []"
      />

      <StrategyPipeline
        :strategy="snapshot.strategy"
        :market="snapshot.market"
        :schedule="snapshot.configuration?.schedule"
      />

      <EquityChart
        :points="curve"
        :baseline="inception?.start_equity"
        :sessions="snapshot.statistics?.sessions"
        :open-pnl="openPerformance.pnl"
        :open-day="snapshot.trading_day"
        :open-positions="openPerformance.positions"
      />

      <div class="grid gap-5 lg:grid-cols-2">
        <PerformanceTable
          :points="curve"
          :as-of="snapshot.trading_day"
        />

        <UCard
          title="Account"
          variant="subtle"
          :ui="{ header: 'p-2 sm:p-2', body: 'p-0 sm:p-0' }"
        >
          <UTable
            :data="accountRows"
            :columns="accountColumns"
          >
            <template #value-cell="{ row }">
              <span
                class="numeric"
                :class="{
                  'font-semibold text-highlighted': row.original.emphasis,
                  'text-muted': row.original.muted
                }"
              >
                {{ row.original.value }}
              </span>
            </template>
          </UTable>
        </UCard>
      </div>

      <PositionsTable :positions="snapshot.positions ?? []" />

      <SessionsTable :sessions="recentSessions" />
    </template>
  </div>
</template>
