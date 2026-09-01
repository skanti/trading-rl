<script setup lang="ts">
import { formatCurrency, formatDay } from '~/utils/format'

const { snapshot, ensureLoaded, pending } = useSnapshot()

await ensureLoaded()

const strategy = computed(() => snapshot.value?.strategy)
const closed = computed(() => snapshot.value?.closed_basket ?? [])
</script>

<template>
  <div class="space-y-5">
    <div
      v-if="pending && !snapshot"
      class="space-y-5"
    >
      <USkeleton class="h-40 rounded-xl" />
      <USkeleton class="h-64 rounded-xl" />
    </div>

    <template v-else-if="snapshot">
      <UCard
        variant="subtle"
      >
        <template #header>
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 class="text-sm font-semibold text-highlighted">
                Current basket
              </h1>
              <p class="text-xs text-muted">
                Equal-notional positions held from the afternoon entry until the next session.
              </p>
            </div>
            <StatusBadge :status="strategy?.status" />
          </div>
        </template>
        <dl class="grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt class="text-xs uppercase tracking-wide text-slate-500">
              Entered
            </dt>
            <dd class="numeric mt-0.5 text-slate-200">
              {{ formatDay(strategy?.entry_date) }}
            </dd>
          </div>
          <div>
            <dt class="text-xs uppercase tracking-wide text-slate-500">
              Exits
            </dt>
            <dd class="numeric mt-0.5 text-slate-200">
              {{ formatDay(strategy?.exit_date) }}
            </dd>
          </div>
          <div>
            <dt class="text-xs uppercase tracking-wide text-slate-500">
              Budget
            </dt>
            <dd class="numeric mt-0.5 text-slate-200">
              {{ strategy?.budget ? formatCurrency(strategy.budget) : '—' }}
            </dd>
            <dd
              v-if="strategy?.estimated_deployed_notional"
              class="numeric mt-0.5 text-xs text-slate-500"
            >
              Est. deployed {{ formatCurrency(strategy.estimated_deployed_notional) }}
            </dd>
          </div>
          <div>
            <dt class="text-xs uppercase tracking-wide text-slate-500">
              Per symbol
            </dt>
            <dd class="numeric mt-0.5 text-slate-200">
              {{ strategy?.per_symbol_notional ? formatCurrency(strategy.per_symbol_notional) : '—' }}
            </dd>
            <dd
              v-if="strategy?.share_mode"
              class="mt-0.5 text-xs capitalize text-slate-500"
            >
              {{ strategy.share_mode }} shares
            </dd>
          </div>
        </dl>

        <UAlert
          v-if="strategy?.remaining_symbols?.length"
          class="mt-2"
          color="warning"
          variant="subtle"
          icon="i-lucide-triangle-alert"
          title="Positions remain after the exit attempt"
          :description="strategy.remaining_symbols.join(', ')"
        />
      </UCard>

      <PositionsTable :positions="snapshot.positions ?? []" />

      <UCard
        title="Most recent closed basket"
        description="Realized per symbol, before fees"
        variant="subtle"
      >
        <ClosedTradesTable
          v-if="closed.length"
          :trades="closed"
          :totals="snapshot.basket_totals"
        />
        <UEmpty
          v-else
          icon="i-lucide-history"
          title="Nothing closed yet"
          description="The first completed basket will appear here."
          class="py-8"
        />
      </UCard>
    </template>
  </div>
</template>
