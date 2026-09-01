<script setup lang="ts">
import {
  formatCurrency,
  formatDay,
  formatSignedCurrency,
  formatSignedPercent,
  toneClass
} from '~/utils/format'
import type { SessionRecord } from '~/types/dashboard'
import type { TableColumn } from '@nuxt/ui'

const props = defineProps<{ sessions: SessionRecord[] }>()
const expanded = ref<Record<string, boolean>>({})

const traded = computed(() => props.sessions.filter(session => session.trades.length > 0))
const realizedTotal = computed(() =>
  traded.value.reduce((sum, session) => sum + (session.realized_pnl ?? 0), 0)
)

const columns: TableColumn<SessionRecord>[] = [
  { id: 'expand', header: '' },
  { accessorKey: 'trading_day', header: 'Day' },
  {
    accessorKey: 'last_action',
    header: 'Action',
    meta: { class: { th: 'hidden sm:table-cell', td: 'hidden sm:table-cell' } }
  },
  {
    accessorKey: 'symbols',
    header: 'Symbols',
    meta: { class: { th: 'hidden sm:table-cell', td: 'hidden sm:table-cell' } }
  },
  {
    accessorKey: 'entry_notional',
    header: 'Deployed',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'realized_pnl',
    header: 'Net P&L',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'realized_return',
    header: 'Return',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  }
]
</script>

<template>
  <UCard
    variant="subtle"
    class="min-w-0 max-w-full"
  >
    <template #header>
      <div class="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 class="text-sm font-semibold text-highlighted">
            Session history
          </h2>
          <p class="text-xs text-muted">
            One row per trading day recorded by the daemon
          </p>
        </div>
        <UBadge
          v-if="traded.length"
          color="neutral"
          variant="subtle"
          class="numeric"
        >
          {{ formatSignedCurrency(realizedTotal) }} · {{ traded.length }} sessions
        </UBadge>
      </div>
    </template>

    <UEmpty
      v-if="!sessions.length"
      icon="i-lucide-history"
      title="No sessions recorded yet"
      description="The daemon writes one summary per trading day."
      class="py-8"
    />

    <UTable
      v-else
      v-model:expanded="expanded"
      :data="sessions"
      :columns="columns"
      :get-row-id="row => row.trading_day"
      :ui="{ tr: 'data-[expanded=true]:bg-elevated/50' }"
    >
      <template #expand-cell="{ row }">
        <UButton
          v-if="row.original.trades.length"
          color="neutral"
          variant="ghost"
          size="xs"
          square
          :icon="row.getIsExpanded() ? 'i-lucide-chevron-down' : 'i-lucide-chevron-right'"
          :aria-label="row.getIsExpanded() ? 'Collapse session' : 'Expand session'"
          @click="row.toggleExpanded()"
        />
      </template>
      <template #trading_day-cell="{ row }">
        <div>
          <span class="numeric font-medium text-highlighted">
            {{ formatDay(row.original.trading_day) }}
          </span>
          <p
            v-if="row.original.error"
            class="mt-0.5 line-clamp-1 text-xs text-error"
          >
            {{ row.original.error }}
          </p>
        </div>
      </template>
      <template #last_action-cell="{ row }">
        <UBadge
          :color="row.original.error ? 'error' : row.original.last_action === 'exit' ? 'primary' : 'neutral'"
          variant="subtle"
          size="sm"
        >
          {{ row.original.error ? 'error' : row.original.last_action ?? '—' }}
        </UBadge>
      </template>
      <template #symbols-cell="{ row }">
        <span class="line-clamp-1 text-muted">
          {{ row.original.symbols.length ? row.original.symbols.join(', ') : '—' }}
        </span>
      </template>
      <template #entry_notional-cell="{ row }">
        <span class="numeric">{{ row.original.entry_notional ? formatCurrency(row.original.entry_notional) : '—' }}</span>
      </template>
      <template #realized_pnl-cell="{ row }">
        <span
          class="numeric font-semibold"
          :class="toneClass(row.original.realized_pnl)"
        >
          {{ row.original.realized_pnl === null ? '—' : formatSignedCurrency(row.original.realized_pnl) }}
        </span>
      </template>
      <template #realized_return-cell="{ row }">
        <span
          class="numeric"
          :class="toneClass(row.original.realized_pnl)"
        >
          {{ row.original.realized_return === null ? '—' : formatSignedPercent(row.original.realized_return) }}
        </span>
      </template>
      <template #expanded="{ row }">
        <ClosedTradesTable
          :trades="row.original.trades"
        />
      </template>
    </UTable>
  </UCard>
</template>
