<script setup lang="ts">
import {
  formatCurrency,
  formatQuantity,
  formatSignedCurrency,
  formatSignedPercent,
  toneClass
} from '~/utils/format'
import type { Position } from '~/types/dashboard'
import type { TableColumn } from '@nuxt/ui'

const props = withDefaults(defineProps<{
  positions: Position[]
  title?: string
  emptyMessage?: string
}>(), {
  title: 'Open positions',
  emptyMessage: 'No open positions. The strategy is flat.'
})

const totals = computed(() => {
  const marketValue = props.positions.reduce((sum, item) => sum + (item.market_value ?? 0), 0)
  const unrealized = props.positions.reduce((sum, item) => sum + (item.unrealized_pl ?? 0), 0)
  const cost = props.positions.reduce((sum, item) => sum + (item.cost_basis ?? 0), 0)
  return { marketValue, unrealized, unrealizedPct: cost ? unrealized / cost : 0 }
})

const columns: TableColumn<Position>[] = [
  { accessorKey: 'symbol', header: 'Symbol' },
  {
    accessorKey: 'qty',
    header: 'Qty',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'avg_entry_price',
    header: 'Entry',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'current_price',
    header: 'Last',
    meta: { class: { th: 'hidden text-right sm:table-cell', td: 'hidden text-right sm:table-cell' } }
  },
  {
    accessorKey: 'market_value',
    header: 'Value',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'unrealized_pl',
    header: 'P&L',
    meta: { class: { th: 'text-right', td: 'text-right' } }
  },
  {
    accessorKey: 'unrealized_plpc',
    header: '%',
    meta: {
      class: {
        th: 'hidden text-right sm:table-cell',
        td: 'hidden text-right sm:table-cell'
      }
    }
  }
]
</script>

<template>
  <UCard
    class="min-w-0 max-w-full"
  >
    <template #header>
      <div class="flex flex-wrap items-baseline justify-between gap-2">
        <h2 class="text-sm font-semibold text-highlighted">
          {{ title }}
        </h2>
        <UBadge
          v-if="positions.length"
          color="neutral"
          variant="subtle"
        >
          {{ positions.length }} symbol{{ positions.length === 1 ? '' : 's' }}
        </UBadge>
      </div>
    </template>

    <UEmpty
      v-if="!positions.length"
      icon="i-lucide-layers"
      title="No open positions"
      :description="emptyMessage"
      class="py-8"
    />

    <UTable
      v-else
      :data="positions"
      :columns="columns"
    >
      <template #symbol-cell="{ row }">
        <div>
          <span class="font-semibold text-highlighted">{{ row.original.symbol }}</span>
          <span class="numeric mt-0.5 block text-xs text-muted sm:hidden">
            {{ formatQuantity(row.original.qty) }} shares
          </span>
        </div>
      </template>
      <template #qty-cell="{ row }">
        <span class="numeric text-muted">{{ formatQuantity(row.original.qty) }}</span>
      </template>
      <template #avg_entry_price-cell="{ row }">
        <span class="numeric">{{ formatCurrency(row.original.avg_entry_price) }}</span>
      </template>
      <template #current_price-cell="{ row }">
        <span class="numeric">{{ formatCurrency(row.original.current_price) }}</span>
      </template>
      <template #market_value-cell="{ row }">
        <span class="numeric">{{ formatCurrency(row.original.market_value) }}</span>
      </template>
      <template #unrealized_pl-cell="{ row }">
        <div class="flex flex-col items-end">
          <span
            class="numeric font-semibold"
            :class="toneClass(row.original.unrealized_pl)"
          >
            {{ formatSignedCurrency(row.original.unrealized_pl) }}
          </span>
          <span
            class="numeric mt-0.5 text-xs sm:hidden"
            :class="toneClass(row.original.unrealized_pl)"
          >
            {{ formatSignedPercent(row.original.unrealized_plpc) }}
          </span>
        </div>
      </template>
      <template #unrealized_plpc-cell="{ row }">
        <span
          class="numeric"
          :class="toneClass(row.original.unrealized_pl)"
        >
          {{ formatSignedPercent(row.original.unrealized_plpc) }}
        </span>
      </template>
    </UTable>

    <template
      v-if="positions.length"
      #footer
    >
      <div class="grid grid-cols-3 gap-3 text-right text-xs">
        <div>
          <p class="text-muted">
            Position value
          </p>
          <p class="numeric mt-0.5 font-semibold text-highlighted">
            {{ formatCurrency(totals.marketValue) }}
          </p>
        </div>
        <div>
          <p class="text-muted">
            Open P&amp;L
          </p>
          <p
            class="numeric mt-0.5 font-semibold"
            :class="toneClass(totals.unrealized)"
          >
            {{ formatSignedCurrency(totals.unrealized) }}
          </p>
        </div>
        <div>
          <p class="text-muted">
            Return
          </p>
          <p
            class="numeric mt-0.5 font-semibold"
            :class="toneClass(totals.unrealized)"
          >
            {{ formatSignedPercent(totals.unrealizedPct) }}
          </p>
        </div>
      </div>
    </template>
  </UCard>
</template>
