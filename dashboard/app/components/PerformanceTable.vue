<script setup lang="ts">
import { formatCurrency, formatDay, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'
import { BUCKET_ORDER, type PerformanceBucket, type BucketKey } from '~/types/dashboard'

const props = defineProps<{ performance?: Record<BucketKey, PerformanceBucket> | null }>()

const rows = computed(() =>
  BUCKET_ORDER.map(key => props.performance?.[key]).filter(Boolean) as PerformanceBucket[]
)
</script>

<template>
  <div class="rounded-xl border border-slate-800 bg-slate-900/50">
    <div class="border-b border-slate-800 px-4 py-3">
      <h2 class="text-sm font-semibold text-white">
        Performance
      </h2>
      <p class="text-xs text-slate-500">
        Account equity, measured from the close before each period opened
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
      <table class="w-full min-w-[34rem] text-sm">
        <thead>
          <tr class="text-xs uppercase tracking-wide text-slate-500">
            <th class="px-4 py-2 text-left font-medium">
              Period
            </th>
            <th class="px-4 py-2 text-right font-medium">
              P&amp;L
            </th>
            <th class="px-4 py-2 text-right font-medium">
              P&amp;L %
            </th>
            <th class="px-4 py-2 text-right font-medium">
              Equity
            </th>
            <th class="px-4 py-2 text-right font-medium">
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
            <td class="whitespace-nowrap px-4 py-2.5 font-medium text-slate-200">
              {{ row.label }}
            </td>
            <td
              class="numeric px-4 py-2.5 text-right font-semibold"
              :class="toneClass(row.pnl)"
            >
              {{ formatSignedCurrency(row.pnl) }}
            </td>
            <td
              class="numeric px-4 py-2.5 text-right"
              :class="toneClass(row.pnl)"
            >
              {{ formatSignedPercent(row.pnl_pct) }}
            </td>
            <!-- Every bucket ends on the same live equity, so show it once. -->
            <td class="numeric whitespace-nowrap px-4 py-2.5 text-right text-slate-300">
              {{ index === 0 ? formatCurrency(row.end_equity) : '' }}
            </td>
            <td class="numeric whitespace-nowrap px-4 py-2.5 text-right text-slate-500">
              {{ formatDay(row.start_day) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
