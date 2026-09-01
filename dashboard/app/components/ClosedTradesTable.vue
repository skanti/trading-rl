<script setup lang="ts">
import {
  formatCurrency,
  formatQuantity,
  formatSignedCurrency,
  formatSignedPercent,
  toneClass
} from '~/utils/format'
import type { BasketTotals, ClosedTrade } from '~/types/dashboard'
import type { TableColumn } from '@nuxt/ui'

const props = withDefaults(defineProps<{
  trades: ClosedTrade[]
  totals?: BasketTotals
}>(), {
  totals: undefined
})

const summary = computed(() => {
  if (props.totals && props.totals.pnl !== undefined) return props.totals
  const entry = props.trades.reduce((sum, trade) => sum + trade.entry_notional, 0)
  const exitTotal = props.trades.reduce((sum, trade) => sum + trade.exit_notional, 0)
  const pnl = props.trades.reduce((sum, trade) => sum + trade.pnl, 0)
  return { entry_notional: entry, exit_notional: exitTotal, pnl, pnl_pct: entry ? pnl / entry : 0 }
})

const columns: TableColumn<ClosedTrade>[] = [
  { accessorKey: 'symbol', header: 'Symbol' },
  {
    accessorKey: 'qty',
    header: 'Qty',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'entry_price',
    header: 'Entry',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'exit_price',
    header: 'Exit',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'pnl',
    header: 'Gross P&L',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'pnl_pct',
    header: '%',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  }
]
</script>

<template>
  <div class="min-w-0">
    <UTable
      :data="trades"
      :columns="columns"
    >
      <template #symbol-cell="{ row }">
        <span class="font-semibold text-highlighted">{{ row.original.symbol }}</span>
      </template>
      <template #qty-cell="{ row }">
        <span class="numeric text-muted">{{ formatQuantity(row.original.qty) }}</span>
      </template>
      <template #entry_price-cell="{ row }">
        <span class="numeric">{{ formatCurrency(row.original.entry_price) }}</span>
      </template>
      <template #exit_price-cell="{ row }">
        <span class="numeric">{{ formatCurrency(row.original.exit_price) }}</span>
      </template>
      <template #pnl-cell="{ row }">
        <span
          class="numeric font-semibold"
          :class="toneClass(row.original.pnl)"
        >
          {{ formatSignedCurrency(row.original.pnl) }}
        </span>
      </template>
      <template #pnl_pct-cell="{ row }">
        <span
          class="numeric"
          :class="toneClass(row.original.pnl)"
        >
          {{ formatSignedPercent(row.original.pnl_pct) }}
        </span>
      </template>
    </UTable>

    <USeparator />
    <div class="grid grid-cols-2 gap-2 px-2 py-2 text-right text-xs sm:grid-cols-4">
      <div>
        <p class="text-muted">
          Deployed
        </p>
        <p class="numeric mt-0.5 font-semibold text-highlighted">
          {{ formatCurrency(summary.entry_notional) }}
        </p>
      </div>
      <div>
        <p class="text-muted">
          Exit value
        </p>
        <p class="numeric mt-0.5 font-semibold text-highlighted">
          {{ formatCurrency(summary.exit_notional) }}
        </p>
      </div>
      <div>
        <p class="text-muted">
          Gross P&amp;L
        </p>
        <p
          class="numeric mt-0.5 font-semibold"
          :class="toneClass(summary.pnl)"
        >
          {{ formatSignedCurrency(summary.pnl) }}
        </p>
      </div>
      <div>
        <p class="text-muted">
          Return
        </p>
        <p
          class="numeric mt-0.5 font-semibold"
          :class="toneClass(summary.pnl)"
        >
          {{ formatSignedPercent(summary.pnl_pct) }}
        </p>
      </div>
    </div>
  </div>
</template>
