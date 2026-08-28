<script setup lang="ts">
import { formatCurrency, formatDay, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'
import type { SessionRecord } from '~/types/dashboard'

const props = defineProps<{ sessions: SessionRecord[] }>()

const expanded = ref<string | null>(null)

function toggle(day: string) {
  expanded.value = expanded.value === day ? null : day
}

// A day the daemon never completed carries an error instead of a result.
const traded = computed(() => props.sessions.filter(session => session.trades.length > 0))
const realizedTotal = computed(() =>
  traded.value.reduce((sum, session) => sum + (session.realized_pnl ?? 0), 0)
)
</script>

<template>
  <div class="min-w-0 max-w-full rounded-xl border border-slate-800 bg-slate-900/50">
    <div class="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800 px-4 py-3">
      <div>
        <h2 class="text-sm font-semibold text-white">
          Session history
        </h2>
        <p class="text-xs text-slate-500">
          One row per trading day recorded by the daemon
        </p>
      </div>
      <span
        v-if="traded.length"
        class="numeric text-xs"
        :class="toneClass(realizedTotal)"
      >
        {{ formatSignedCurrency(realizedTotal) }} realized across {{ traded.length }} session{{ traded.length === 1 ? '' : 's' }}
      </span>
    </div>

    <div
      v-if="!sessions.length"
      class="px-4 py-6 text-sm text-slate-500"
    >
      No sessions recorded yet. The daemon writes one summary per trading day.
    </div>

    <div
      v-else
      class="scroll-x"
    >
      <table class="w-full table-fixed text-sm sm:min-w-[44rem] sm:table-auto">
        <thead>
          <tr class="text-xs uppercase tracking-wide text-slate-500">
            <th class="w-[42%] px-3 py-2 text-left font-medium sm:w-auto sm:px-4">
              Day
            </th>
            <th class="hidden px-4 py-2 text-left font-medium sm:table-cell">
              Action
            </th>
            <th class="hidden px-4 py-2 text-left font-medium sm:table-cell">
              Symbols
            </th>
            <th class="hidden px-4 py-2 text-right font-medium sm:table-cell">
              Deployed
            </th>
            <th class="px-2 py-2 text-right font-medium sm:px-4">
              Net P&amp;L
            </th>
            <th class="px-3 py-2 text-right font-medium sm:px-4">
              Return
            </th>
          </tr>
        </thead>
        <tbody>
          <template
            v-for="session in sessions"
            :key="session.trading_day"
          >
            <tr
              class="cursor-pointer border-t border-slate-800/70 transition hover:bg-slate-800/40"
              @click="toggle(session.trading_day)"
            >
              <td class="numeric px-3 py-2.5 font-medium text-slate-200 sm:px-4">
                <span class="inline-flex items-center gap-1.5">
                  <UIcon
                    v-if="session.trades.length"
                    :name="expanded === session.trading_day ? 'i-lucide-chevron-down' : 'i-lucide-chevron-right'"
                    class="size-3.5 text-slate-500"
                  />
                  <span
                    v-else
                    class="inline-block size-3.5"
                  />
                  {{ formatDay(session.trading_day) }}
                </span>
              </td>
              <td class="hidden px-4 py-2.5 sm:table-cell">
                <UBadge
                  :color="session.error ? 'error' : session.last_action === 'exit' ? 'primary' : 'neutral'"
                  variant="subtle"
                  size="sm"
                >
                  {{ session.error ? 'error' : session.last_action ?? '—' }}
                </UBadge>
              </td>
              <td class="hidden px-4 py-2.5 text-slate-400 sm:table-cell">
                <span
                  v-if="session.symbols.length"
                  class="line-clamp-1"
                >
                  {{ session.symbols.join(', ') }}
                </span>
                <span v-else>—</span>
              </td>
              <td class="numeric hidden px-4 py-2.5 text-right text-slate-300 sm:table-cell">
                {{ session.entry_notional ? formatCurrency(session.entry_notional) : '—' }}
              </td>
              <td
                class="numeric px-2 py-2.5 text-right font-semibold sm:px-4"
                :class="toneClass(session.realized_pnl)"
              >
                {{ session.realized_pnl === null ? '—' : formatSignedCurrency(session.realized_pnl) }}
              </td>
              <td
                class="numeric px-3 py-2.5 text-right sm:px-4"
                :class="toneClass(session.realized_pnl)"
              >
                {{ session.realized_return === null ? '—' : formatSignedPercent(session.realized_return) }}
              </td>
            </tr>

            <tr
              v-if="expanded === session.trading_day"
              class="border-t border-slate-800/70 bg-slate-950/60"
            >
              <td
                colspan="6"
                class="px-0 py-0"
              >
                <ClosedTradesTable
                  v-if="session.trades.length"
                  :trades="session.trades"
                  dense
                />
                <p
                  v-else
                  class="px-4 py-3 text-xs text-slate-500"
                >
                  {{ session.error ?? 'No fills recorded for this session.' }}
                </p>
              </td>
            </tr>

            <tr
              v-if="session.error && expanded !== session.trading_day"
              class="border-0"
            >
              <td
                colspan="6"
                class="px-4 pb-2.5 text-xs text-rose-400/80"
              >
                <span class="line-clamp-1">{{ session.error }}</span>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </div>
</template>
