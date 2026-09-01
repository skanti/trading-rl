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
const realizedSessions = computed(() => traded.value.filter(
  session => session.realized_pnl !== null && session.fee_status === 'confirmed'
))
const realizedTotal = computed(() =>
  realizedSessions.value.reduce((sum, session) => sum + (session.realized_pnl ?? 0), 0)
)

function feeStatusLabel(session: SessionRecord) {
  if (session.fee_status === 'confirmed') return 'Fees confirmed'
  if (session.fee_status === 'pending') return 'Fees pending'
  if (session.fee_status === 'unavailable') return 'Fees unavailable'
  return 'No fees expected'
}

function feeStatusColor(session: SessionRecord) {
  if (session.fee_status === 'confirmed') return 'success' as const
  if (session.fee_status === 'pending') return 'warning' as const
  if (session.fee_status === 'unavailable') return 'error' as const
  return 'neutral' as const
}

function netLabel(session: SessionRecord) {
  return session.fee_status === 'confirmed' ? 'Strategy net' : 'Provisional net'
}

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
          v-if="realizedSessions.length"
          color="neutral"
          variant="subtle"
          class="numeric"
        >
          {{ formatSignedCurrency(realizedTotal) }} · {{ realizedSessions.length }} confirmed
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
        <div>
          <span
            class="numeric font-semibold"
            :class="toneClass(row.original.realized_pnl)"
          >
            {{ row.original.realized_pnl === null ? '—' : formatSignedCurrency(row.original.realized_pnl) }}
          </span>
          <p
            v-if="row.original.status === 'closed' && row.original.fee_status !== 'confirmed'"
            class="text-xs text-warning"
          >
            provisional · {{ row.original.fee_status === 'pending' ? 'fees pending' : 'fees unavailable' }}
          </p>
        </div>
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
        <div class="bg-elevated/40 px-2 py-2">
          <div class="mb-2 flex items-center justify-between gap-2">
            <span class="text-xs font-medium text-highlighted">P&amp;L reconciliation</span>
            <UBadge
              :color="feeStatusColor(row.original)"
              variant="subtle"
              size="sm"
            >
              {{ feeStatusLabel(row.original) }}
            </UBadge>
          </div>
          <dl class="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-5">
            <div>
              <dt class="text-xs text-muted">
                Gross fill P&amp;L
              </dt>
              <dd
                class="numeric text-sm"
                :class="toneClass(row.original.gross_realized_pnl)"
              >
                {{ row.original.gross_realized_pnl === null ? '—' : formatSignedCurrency(row.original.gross_realized_pnl) }}
              </dd>
            </div>
            <div>
              <dt class="text-xs text-muted">
                Alpaca fees
              </dt>
              <dd class="numeric text-sm">
                {{ row.original.fee_cost === null ? '—' : formatCurrency(row.original.fee_cost) }}
              </dd>
            </div>
            <div>
              <dt class="text-xs text-muted">
                {{ netLabel(row.original) }}
              </dt>
              <dd
                class="numeric text-sm font-medium"
                :class="toneClass(row.original.realized_pnl)"
              >
                {{ row.original.realized_pnl === null ? '—' : formatSignedCurrency(row.original.realized_pnl) }}
              </dd>
            </div>
            <div>
              <dt class="text-xs text-muted">
                Account change
              </dt>
              <dd
                class="numeric text-sm"
                :class="toneClass(row.original.account_equity_change)"
              >
                {{ row.original.account_equity_change === null ? '—' : formatSignedCurrency(row.original.account_equity_change) }}
              </dd>
            </div>
            <div>
              <dt class="text-xs text-muted">
                Unexplained residual
              </dt>
              <dd
                class="numeric text-sm"
                :class="toneClass(row.original.unexplained_residual)"
              >
                {{ row.original.unexplained_residual === null ? '—' : formatSignedCurrency(row.original.unexplained_residual) }}
              </dd>
            </div>
          </dl>
        </div>
        <ClosedTradesTable
          :trades="row.original.trades"
        />
      </template>
    </UTable>
  </UCard>
</template>
