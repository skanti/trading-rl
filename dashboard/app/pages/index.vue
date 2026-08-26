<script setup lang="ts">
import { formatCurrency, formatDay, formatPercent, formatSignedCurrency, formatSignedPercent, toneClass } from '~/utils/format'

const { snapshot, sessions, pending, error, ensureLoaded } = useSnapshot()

await ensureLoaded()

const account = computed(() => snapshot.value?.account ?? {})
const stats = computed(() => snapshot.value?.statistics)
const today = computed(() => snapshot.value?.performance?.today)
const inception = computed(() => snapshot.value?.performance?.inception)
const curve = computed(() => snapshot.value?.equity_curve ?? [])

const recentSessions = computed(() => sessions.value.slice(0, 8))
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
      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <USkeleton
          v-for="index in 4"
          :key="index"
          class="h-24 rounded-xl"
        />
      </div>
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
      <div class="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p class="text-xs uppercase tracking-wide text-slate-500">
            Account equity
          </p>
          <p class="numeric text-3xl font-semibold text-white">
            {{ formatCurrency(account.equity) }}
          </p>
        </div>
        <div class="flex items-center gap-2">
          <StatusBadge :status="snapshot.strategy?.status" />
          <UBadge
            :color="snapshot.market?.is_open ? 'primary' : 'neutral'"
            variant="subtle"
            size="sm"
          >
            {{ snapshot.market?.is_open ? 'Market open' : 'Market closed' }}
          </UBadge>
        </div>
      </div>

      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Today"
          icon="i-lucide-calendar"
          :value="formatSignedCurrency(today?.pnl)"
          :tone="toneClass(today?.pnl)"
          :hint="formatSignedPercent(today?.pnl_pct)"
        />
        <StatTile
          label="Since inception"
          icon="i-lucide-trending-up"
          :value="formatSignedCurrency(inception?.pnl)"
          :tone="toneClass(inception?.pnl)"
          :hint="`${formatSignedPercent(inception?.pnl_pct)} over ${stats?.sessions ?? 0} sessions`"
        />
        <StatTile
          label="Max drawdown"
          icon="i-lucide-trending-down"
          :value="formatCurrency(stats?.max_drawdown)"
          :tone="(stats?.max_drawdown ?? 0) > 0 ? 'text-rose-400' : 'text-slate-400'"
          :hint="formatPercent(stats?.max_drawdown_pct)"
        />
        <StatTile
          label="Win rate"
          icon="i-lucide-target"
          :value="formatPercent(stats?.win_rate, 1)"
          :hint="`${stats?.winning_sessions ?? 0}W / ${stats?.losing_sessions ?? 0}L`"
        />
      </div>

      <EquityChart
        :points="curve"
        :baseline="inception?.start_equity"
      />

      <div class="grid gap-5 lg:grid-cols-2">
        <PerformanceTable :performance="snapshot.performance" />

        <div class="rounded-xl border border-slate-800 bg-slate-900/50">
          <div class="border-b border-slate-800 px-4 py-3">
            <h2 class="text-sm font-semibold text-white">
              Account
            </h2>
          </div>
          <dl class="divide-y divide-slate-800/70 text-sm">
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Cash
              </dt>
              <dd class="numeric text-slate-200">
                {{ formatCurrency(account.cash) }}
              </dd>
            </div>
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Long market value
              </dt>
              <dd class="numeric text-slate-200">
                {{ formatCurrency(account.long_market_value) }}
              </dd>
            </div>
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Buying power
              </dt>
              <dd class="numeric text-slate-200">
                {{ formatCurrency(account.buying_power) }}
              </dd>
            </div>
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Best session
              </dt>
              <dd
                class="numeric"
                :class="toneClass(stats?.best_day?.profit_loss)"
              >
                {{ stats?.best_day ? `${formatSignedCurrency(stats.best_day.profit_loss)} · ${formatDay(stats.best_day.day)}` : '—' }}
              </dd>
            </div>
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Worst session
              </dt>
              <dd
                class="numeric"
                :class="toneClass(stats?.worst_day?.profit_loss)"
              >
                {{ stats?.worst_day ? `${formatSignedCurrency(stats.worst_day.profit_loss)} · ${formatDay(stats.worst_day.day)}` : '—' }}
              </dd>
            </div>
            <div class="flex justify-between px-4 py-2.5">
              <dt class="text-slate-400">
                Account
              </dt>
              <dd class="numeric text-slate-500">
                {{ account.account_number ?? '—' }}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <PositionsTable :positions="snapshot.positions ?? []" />

      <SessionsTable :sessions="recentSessions" />
    </template>
  </div>
</template>
