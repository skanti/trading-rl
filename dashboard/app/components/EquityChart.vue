<script setup lang="ts">
import { areaPath, buildScale, gridValues, linePath, tickIndices } from '~/utils/chart'
import { formatCompactCurrency, formatCurrency, formatDay, formatSignedCurrency, formatSignedPercent } from '~/utils/format'
import type { EquityPoint } from '~/types/dashboard'

const props = withDefaults(defineProps<{
  points: EquityPoint[]
  baseline?: number
  height?: number
}>(), {
  baseline: undefined,
  height: 300
})

const WIDTH = 960
const PADDING = { top: 16, right: 16, bottom: 28, left: 68 }

const geometry = computed(() => ({ width: WIDTH, height: props.height, padding: PADDING }))
const scale = computed(() => buildScale(props.points, geometry.value))

const anchor = computed(() => props.baseline ?? props.points[0]?.equity ?? 0)
const latest = computed(() => props.points[props.points.length - 1]?.equity ?? 0)
const gaining = computed(() => latest.value >= anchor.value)

const line = computed(() => linePath(props.points, scale.value))
const area = computed(() => areaPath(props.points, scale.value, props.height - PADDING.bottom))

/** The inception line only renders when it actually falls inside the visible domain. */
const baselineY = computed(() => {
  const value = anchor.value
  return value >= scale.value.min && value <= scale.value.max ? scale.value.y(value) : null
})

const grid = computed(() => gridValues(scale.value, 4).map(value => ({
  value,
  y: scale.value.y(value)
})))

const ticks = computed(() => tickIndices(props.points.length, 6).map(index => ({
  index,
  x: scale.value.x(index),
  label: props.points[index]?.day ?? ''
})))

const hovered = ref<number | null>(null)

const active = computed(() => {
  if (hovered.value === null) return null
  const point = props.points[hovered.value]
  if (!point) return null
  return {
    point,
    x: scale.value.x(hovered.value),
    y: scale.value.y(point.equity),
    change: point.equity - anchor.value,
    changePct: anchor.value ? (point.equity - anchor.value) / anchor.value : 0
  }
})

function onMove(event: MouseEvent) {
  if (!props.points.length) return
  const target = event.currentTarget as SVGSVGElement
  const bounds = target.getBoundingClientRect()
  // Map the pointer back through the viewBox scale to a data index.
  const ratio = (event.clientX - bounds.left) / bounds.width
  const x = ratio * WIDTH
  const span = Math.max(1, props.points.length - 1)
  const index = Math.round(((x - PADDING.left) / scale.value.innerWidth) * span)
  hovered.value = Math.min(props.points.length - 1, Math.max(0, index))
}

const dayLabel = (day: string | null | undefined) => formatDay(day)
</script>

<template>
  <div class="relative rounded-xl border border-slate-800 bg-slate-900/50 p-4">
    <div
      v-if="!points.length"
      class="flex h-48 items-center justify-center text-sm text-slate-500"
    >
      No equity history published yet.
    </div>

    <template v-else>
      <div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p class="text-xs font-medium uppercase tracking-wide text-slate-400">
            Equity curve · {{ points.length }} sessions
          </p>
          <p class="numeric mt-1 text-2xl font-semibold text-white">
            {{ formatCurrency(active?.point.equity ?? latest) }}
          </p>
        </div>
        <div class="text-right">
          <p
            class="numeric text-sm font-semibold"
            :class="(active?.change ?? (latest - anchor)) >= 0 ? 'text-emerald-400' : 'text-rose-400'"
          >
            {{ formatSignedCurrency(active?.change ?? (latest - anchor)) }}
            ({{ formatSignedPercent(active?.changePct ?? (anchor ? (latest - anchor) / anchor : 0)) }})
          </p>
          <p class="numeric text-xs text-slate-500">
            <!-- Without a hover this is the last close, which can differ from live
                 account equity; say so rather than showing two unexplained figures. -->
            {{ active
              ? dayLabel(active.point.day)
              : `Last close · ${dayLabel(points[points.length - 1]?.day)}` }}
          </p>
        </div>
      </div>

      <svg
        :viewBox="`0 0 ${WIDTH} ${height}`"
        class="w-full"
        :style="{ height: `${height}px` }"
        preserveAspectRatio="none"
        role="img"
        aria-label="Account equity over time"
        @mousemove="onMove"
        @mouseleave="hovered = null"
      >
        <defs>
          <linearGradient
            :id="gaining ? 'equity-up' : 'equity-down'"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="0%"
              :stop-color="gaining ? '#10b981' : '#f43f5e'"
              stop-opacity="0.28"
            />
            <stop
              offset="100%"
              :stop-color="gaining ? '#10b981' : '#f43f5e'"
              stop-opacity="0"
            />
          </linearGradient>
        </defs>

        <g>
          <line
            v-for="row in grid"
            :key="`grid-${row.value}`"
            :x1="PADDING.left"
            :x2="WIDTH - PADDING.right"
            :y1="row.y"
            :y2="row.y"
            stroke="#1e293b"
            stroke-width="1"
          />
          <text
            v-for="row in grid"
            :key="`label-${row.value}`"
            :x="PADDING.left - 10"
            :y="row.y + 4"
            text-anchor="end"
            fill="#64748b"
            font-size="11"
          >{{ formatCompactCurrency(row.value) }}</text>
        </g>

        <line
          v-if="baselineY !== null"
          :x1="PADDING.left"
          :x2="WIDTH - PADDING.right"
          :y1="baselineY"
          :y2="baselineY"
          stroke="#475569"
          stroke-width="1"
          stroke-dasharray="4 4"
        />

        <path
          :d="area"
          :fill="`url(#${gaining ? 'equity-up' : 'equity-down'})`"
        />
        <path
          :d="line"
          fill="none"
          :stroke="gaining ? '#34d399' : '#fb7185'"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
          vector-effect="non-scaling-stroke"
        />

        <g v-if="active">
          <line
            :x1="active.x"
            :x2="active.x"
            :y1="PADDING.top"
            :y2="height - PADDING.bottom"
            stroke="#64748b"
            stroke-width="1"
            stroke-dasharray="3 3"
          />
          <circle
            :cx="active.x"
            :cy="active.y"
            r="4"
            :fill="gaining ? '#34d399' : '#fb7185'"
            stroke="#020617"
            stroke-width="2"
          />
        </g>

        <text
          v-for="tick in ticks"
          :key="`tick-${tick.index}`"
          :x="tick.x"
          :y="height - 8"
          text-anchor="middle"
          fill="#64748b"
          font-size="11"
        >{{ tick.label.slice(5) }}</text>
      </svg>
    </template>
  </div>
</template>
