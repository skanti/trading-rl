<script setup lang="ts">
import { formatCurrency, formatQuantity, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'
import type { BasketTotals, ClosedTrade } from '~/types/dashboard'

const props = withDefaults(defineProps<{
  trades: ClosedTrade[]
  totals?: BasketTotals
  dense?: boolean
}>(), {
  totals: undefined,
  dense: false
})

// Fall back to summing the rows when the publisher had no totals to write.
const summary = computed(() => {
  if (props.totals && props.totals.pnl !== undefined) return props.totals
  const entry = props.trades.reduce((sum, trade) => sum + trade.entry_notional, 0)
  const exitTotal = props.trades.reduce((sum, trade) => sum + trade.exit_notional, 0)
  const pnl = props.trades.reduce((sum, trade) => sum + trade.pnl, 0)
  return { entry_notional: entry, exit_notional: exitTotal, pnl, pnl_pct: entry ? pnl / entry : 0 }
})
</script>

<template>
  <div class="scroll-x">
    <table class="w-full min-w-[38rem] text-sm">
      <thead>
        <tr class="text-xs uppercase tracking-wide text-slate-500">
          <th class="px-4 py-2 text-left font-medium">
            Symbol
          </th>
          <th class="px-4 py-2 text-right font-medium">
            Qty
          </th>
          <th class="px-4 py-2 text-right font-medium">
            Entry
          </th>
          <th class="px-4 py-2 text-right font-medium">
            Exit
          </th>
          <th class="px-4 py-2 text-right font-medium">
            P&amp;L
          </th>
          <th class="px-4 py-2 text-right font-medium">
            %
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="trade in trades"
          :key="trade.symbol"
          class="border-t border-slate-800/70"
        >
          <td
            :class="dense ? 'px-4 py-1.5' : 'px-4 py-2.5'"
            class="font-semibold text-slate-100"
          >
            {{ trade.symbol }}
          </td>
          <td
            class="numeric px-4 text-right text-slate-400"
            :class="dense ? 'py-1.5' : 'py-2.5'"
          >
            {{ formatQuantity(trade.qty) }}
          </td>
          <td
            class="numeric px-4 text-right text-slate-300"
            :class="dense ? 'py-1.5' : 'py-2.5'"
          >
            {{ formatCurrency(trade.entry_price) }}
          </td>
          <td
            class="numeric px-4 text-right text-slate-300"
            :class="dense ? 'py-1.5' : 'py-2.5'"
          >
            {{ formatCurrency(trade.exit_price) }}
          </td>
          <td
            class="numeric px-4 text-right font-semibold"
            :class="[toneClass(trade.pnl), dense ? 'py-1.5' : 'py-2.5']"
          >
            {{ formatSignedCurrency(trade.pnl) }}
          </td>
          <td
            class="numeric px-4 text-right"
            :class="[toneClass(trade.pnl), dense ? 'py-1.5' : 'py-2.5']"
          >
            {{ formatSignedPercent(trade.pnl_pct) }}
          </td>
        </tr>
      </tbody>
      <tfoot>
        <!-- Notional totals sit under the columns they belong to: deployed under
             Entry, realised under Exit. -->
        <tr class="border-t border-slate-700">
          <td
            class="px-4 py-2.5 font-semibold text-slate-200"
            colspan="2"
          >
            Total
          </td>
          <td class="numeric px-4 py-2.5 text-right text-slate-400">
            {{ formatCurrency(summary.entry_notional) }}
          </td>
          <td class="numeric px-4 py-2.5 text-right text-slate-400">
            {{ formatCurrency(summary.exit_notional) }}
          </td>
          <td
            class="numeric px-4 py-2.5 text-right font-semibold"
            :class="toneClass(summary.pnl ?? 0)"
          >
            {{ formatSignedCurrency(summary.pnl ?? 0) }}
          </td>
          <td
            class="numeric px-4 py-2.5 text-right"
            :class="toneClass(summary.pnl ?? 0)"
          >
            {{ formatSignedPercent(summary.pnl_pct ?? 0) }}
          </td>
        </tr>
      </tfoot>
    </table>
  </div>
</template>
