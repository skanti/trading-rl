<script setup lang="ts">
import { formatCurrency, formatQuantity, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'
import type { Position } from '~/types/dashboard'

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
</script>

<template>
  <div class="min-w-0 max-w-full rounded-xl border border-slate-800 bg-slate-900/50">
    <div class="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800 px-4 py-3">
      <h2 class="text-sm font-semibold text-white">
        {{ title }}
      </h2>
      <span
        v-if="positions.length"
        class="numeric text-xs text-slate-500"
      >
        {{ positions.length }} symbol{{ positions.length === 1 ? '' : 's' }}
        · {{ formatCurrency(totals.marketValue) }}
      </span>
    </div>

    <div
      v-if="!positions.length"
      class="px-4 py-6 text-sm text-slate-500"
    >
      {{ emptyMessage }}
    </div>

    <div
      v-else
      class="scroll-x"
    >
      <table class="w-full table-fixed text-sm sm:min-w-[42rem] sm:table-auto">
        <thead>
          <tr class="text-xs uppercase tracking-wide text-slate-500">
            <th class="w-[22%] px-3 py-2 text-left font-medium sm:w-auto sm:px-4">
              Symbol
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              Qty
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              Entry
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              Last
            </th>
            <th class="px-2 py-2 text-right font-medium sm:px-4">
              <span class="sm:hidden">Value</span>
              <span class="hidden sm:inline">Market value</span>
            </th>
            <th class="px-2 py-2 text-right font-medium sm:px-4">
              <span class="sm:hidden">P&amp;L</span>
              <span class="hidden sm:inline">Unrealized</span>
            </th>
            <th class="px-3 py-2 text-right font-medium sm:px-4">
              %
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="position in positions"
            :key="position.symbol"
            class="border-t border-slate-800/70"
          >
            <td class="px-3 py-2.5 font-semibold text-slate-100 sm:px-4">
              {{ position.symbol }}
            </td>
            <td class="numeric hidden px-4 py-2.5 text-right text-slate-400 sm:table-cell">
              {{ formatQuantity(position.qty) }}
            </td>
            <td class="numeric hidden px-4 py-2.5 text-right text-slate-300 sm:table-cell">
              {{ formatCurrency(position.avg_entry_price) }}
            </td>
            <td class="numeric hidden px-4 py-2.5 text-right text-slate-300 sm:table-cell">
              {{ formatCurrency(position.current_price) }}
            </td>
            <td class="numeric px-2 py-2.5 text-right text-slate-300 sm:px-4">
              {{ formatCurrency(position.market_value) }}
            </td>
            <td
              class="numeric px-2 py-2.5 text-right font-semibold sm:px-4"
              :class="toneClass(position.unrealized_pl)"
            >
              {{ formatSignedCurrency(position.unrealized_pl) }}
            </td>
            <td
              class="numeric px-3 py-2.5 text-right sm:px-4"
              :class="toneClass(position.unrealized_pl)"
            >
              {{ formatSignedPercent(position.unrealized_plpc) }}
            </td>
          </tr>
        </tbody>
        <tfoot>
          <tr class="border-t border-slate-700 sm:hidden">
            <td class="px-3 py-2.5 font-semibold text-slate-200">
              Total
            </td>
            <td class="numeric px-2 py-2.5 text-right text-slate-200">
              {{ formatCurrency(totals.marketValue) }}
            </td>
            <td
              class="numeric px-2 py-2.5 text-right font-semibold"
              :class="toneClass(totals.unrealized)"
            >
              {{ formatSignedCurrency(totals.unrealized) }}
            </td>
            <td
              class="numeric px-3 py-2.5 text-right"
              :class="toneClass(totals.unrealized)"
            >
              {{ formatSignedPercent(totals.unrealizedPct) }}
            </td>
          </tr>
          <tr class="hidden border-t border-slate-700 sm:table-row">
            <td
              class="px-4 py-2.5 font-semibold text-slate-200"
              colspan="4"
            >
              Total
            </td>
            <td class="numeric px-4 py-2.5 text-right text-slate-200">
              {{ formatCurrency(totals.marketValue) }}
            </td>
            <td
              class="numeric px-4 py-2.5 text-right font-semibold"
              :class="toneClass(totals.unrealized)"
            >
              {{ formatSignedCurrency(totals.unrealized) }}
            </td>
            <td
              class="numeric px-4 py-2.5 text-right"
              :class="toneClass(totals.unrealized)"
            >
              {{ formatSignedPercent(totals.unrealizedPct) }}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  </div>
</template>
