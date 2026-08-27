<script setup lang="ts">
import { formatCurrency, formatDay, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'
import { BUCKET_ORDER, type PerformanceBucket, type BucketKey } from '~/types/dashboard'

const props = defineProps<{ performance?: Record<BucketKey, PerformanceBucket> | null }>()

const rows = computed(() =>
  BUCKET_ORDER.map(key => props.performance?.[key]).filter(Boolean) as PerformanceBucket[]
)
</script>

<template>
  <div class="min-w-0 max-w-full rounded-xl border border-slate-800 bg-slate-900/50">
    <div class="border-b border-slate-800 px-4 py-3">
      <h2 class="text-sm font-semibold text-white">
        Performance
      </h2>
      <p class="text-xs text-slate-500">
        Open positions are live; longer periods use closed sessions
      </p>
    </div>

    <div
      v-if="!rows.length"
      class="px-4 py-6 text-sm text-slate-500"
    >
      No performance published yet.
    </div>

    <div
      v-else
      class="scroll-x"
    >
      <table class="w-full table-fixed text-sm sm:min-w-[34rem] sm:table-auto">
        <thead>
          <tr class="text-xs uppercase tracking-wide text-slate-500">
            <th class="w-[42%] px-3 py-2 text-left font-medium sm:w-auto sm:px-4">
              Period
            </th>
            <th class="px-2 py-2 text-right font-medium sm:px-4">
              P&amp;L
            </th>
            <th class="px-3 py-2 text-right font-medium sm:px-4">
              P&amp;L %
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              Equity
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              From
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, index) in rows"
            :key="row.key"
            class="border-t border-slate-800/70"
          >
            <td class="whitespace-nowrap px-3 py-2.5 font-medium text-slate-200 sm:px-4">
              {{ row.label }}
            </td>
            <td
              class="numeric px-2 py-2.5 text-right font-semibold sm:px-4"
              :class="toneClass(row.pnl)"
            >
              {{ formatSignedCurrency(row.pnl) }}
            </td>
            <td
              class="numeric px-3 py-2.5 text-right sm:px-4"
              :class="toneClass(row.pnl)"
            >
              {{ formatSignedPercent(row.pnl_pct) }}
            </td>
            <!-- The first row carries current account equity; the remaining rows omit
                 their realized-curve endpoint to keep the table compact. -->
            <td class="numeric hidden whitespace-nowrap px-4 py-2.5 text-right text-slate-300 sm:table-cell">
              {{ index === 0 ? formatCurrency(row.end_equity) : '' }}
            </td>
            <td class="numeric hidden whitespace-nowrap px-4 py-2.5 text-right text-slate-500 sm:table-cell">
              {{ formatDay(row.start_day) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
