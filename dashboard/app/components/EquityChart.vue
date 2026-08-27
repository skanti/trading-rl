<script setup lang="ts">
import { areaPath, buildScale, gridValues, linePath, tickIndices } from '~/utils/chart'
import { formatAxisCurrency, formatCurrency, formatDay, formatSignedCurrency, formatSignedPercent } from '~/utils/format'
import type { EquityPoint } from '~/types/dashboard'

const props = withDefaults(defineProps<{
  points: EquityPoint[]
  baseline?: number
  height?: number
  sessions?: number
  openPnl?: number
  openDay?: string
  openPositions?: number
}>(), {
  baseline: undefined,
  height: 300,
  sessions: undefined,
  openPnl: 0,
  openDay: undefined,
  openPositions: 0
})

const DEFAULT_WIDTH = 960
const chartHost = useTemplateRef<HTMLElement>('chartHost')
const chartWidth = ref(DEFAULT_WIDTH)
let resizeObserver: ResizeObserver | undefined

onMounted(() => {
  const updateWidth = () => {
    chartWidth.value = Math.max(260, Math.round(chartHost.value?.clientWidth ?? DEFAULT_WIDTH))
  }
  updateWidth()
  resizeObserver = new ResizeObserver(updateWidth)
  if (chartHost.value) resizeObserver.observe(chartHost.value)
})
onBeforeUnmount(() => resizeObserver?.disconnect())

const compact = computed(() => chartWidth.value < 520)
const geometry = computed(() => ({
  width: chartWidth.value,
  height: props.height,
  padding: compact.value
    ? { top: 10, right: 8, bottom: 24, left: 50 }
    : { top: 12, right: 12, bottom: 26, left: 60 }
}))

const anchor = computed(() => props.baseline ?? props.points[0]?.equity ?? 0)
const realizedLatest = computed(() => props.points[props.points.length - 1]?.equity ?? 0)
const openPoint = computed<EquityPoint | null>(() => {
  if (!props.openPositions || !props.points.length) return null
  const pnl = Number.isFinite(props.openPnl) ? props.openPnl : 0
  const equity = realizedLatest.value + pnl
  return {
    day: props.openDay ?? props.points[props.points.length - 1]!.day,
    equity,
    profit_loss: pnl,
    profit_loss_pct: realizedLatest.value ? pnl / realizedLatest.value : 0
  }
})
const displayPoints = computed(() => openPoint.value
  ? [...props.points, openPoint.value]
  : props.points)
const scale = computed(() => buildScale(displayPoints.value, geometry.value))
const latest = computed(() => displayPoints.value[displayPoints.value.length - 1]?.equity ?? 0)
const gaining = computed(() => realizedLatest.value >= anchor.value)
const sessionCount = computed(() => props.sessions ?? Math.max(0, props.points.length - 1))

const line = computed(() => {
  if (openPoint.value && props.points.length === 1) {
    const point = props.points[0]!
    return `M ${scale.value.x(0)} ${scale.value.y(point.equity)}`
  }
  return linePath(props.points, scale.value)
})
const area = computed(() => openPoint.value && props.points.length === 1
  ? ''
  : areaPath(props.points, scale.value, props.height - geometry.value.padding.bottom))
const openSegment = computed(() => {
  if (!openPoint.value || !props.points.length) return ''
  const realizedIndex = props.points.length - 1
  return [
    `M ${scale.value.x(realizedIndex)} ${scale.value.y(realizedLatest.value)}`,
    `L ${scale.value.x(realizedIndex + 1)} ${scale.value.y(openPoint.value.equity)}`
  ].join(' ')
})

/** The inception line only renders when it actually falls inside the visible domain. */
const baselineY = computed(() => {
  const value = anchor.value
  return value >= scale.value.min && value <= scale.value.max ? scale.value.y(value) : null
})

const grid = computed(() => gridValues(scale.value, compact.value ? 3 : 4).map(value => ({
  value,
  y: scale.value.y(value)
})))

const ticks = computed(() => {
  const indices = tickIndices(displayPoints.value.length, compact.value ? 2 : chartWidth.value < 760 ? 3 : 6)
  return indices.map((index, position) => ({
    index,
    x: scale.value.x(index),
    label: shortDay(displayPoints.value[index]?.day),
    anchor: position === 0 ? 'start' : position === indices.length - 1 ? 'end' : 'middle'
  }))
})

const hovered = ref<number | null>(null)

const active = computed(() => {
  if (hovered.value === null) return null
  const point = displayPoints.value[hovered.value]
  if (!point) return null
  return {
    point,
    open: Boolean(openPoint.value && hovered.value === displayPoints.value.length - 1),
    x: scale.value.x(hovered.value),
    y: scale.value.y(point.equity),
    change: point.equity - anchor.value,
    changePct: anchor.value ? (point.equity - anchor.value) / anchor.value : 0
  }
})

function onMove(event: PointerEvent) {
  if (!displayPoints.value.length) return
  const target = event.currentTarget as SVGSVGElement
  const bounds = target.getBoundingClientRect()
  // Map the pointer back through the viewBox scale to a data index.
  const ratio = (event.clientX - bounds.left) / bounds.width
  const x = ratio * geometry.value.width
  const span = Math.max(1, displayPoints.value.length - 1)
  const index = Math.round(
    ((x - geometry.value.padding.left) / scale.value.innerWidth) * span
  )
  hovered.value = Math.min(displayPoints.value.length - 1, Math.max(0, index))
}

const dayLabel = (day: string | null | undefined) => formatDay(day)

function shortDay(day: string | null | undefined): string {
  if (!day) return ''
  const parsed = new Date(`${day}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) return day
  return parsed.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}
</script>

<template>
  <div
    class="equity-chart relative select-none rounded-xl border border-slate-800 bg-slate-900/50 p-3 sm:p-4"
    @selectstart.prevent
  >
    <div
      v-if="!points.length"
      class="flex h-48 items-center justify-center text-sm text-slate-500"
    >
      No equity history published yet.
    </div>

    <template v-else>
      <div class="mb-3">
        <p class="text-xs font-medium uppercase tracking-wide text-slate-400">
          Strategy equity · {{ sessionCount }} closed session{{ sessionCount === 1 ? '' : 's' }}
        </p>
        <div class="mt-1 flex items-baseline justify-between gap-2">
          <p class="numeric whitespace-nowrap text-[clamp(0.875rem,4vw,1.5rem)] font-semibold text-white">
            {{ formatCurrency(active?.point.equity ?? latest) }}
          </p>
          <div class="min-w-0 text-right">
            <p
              class="numeric whitespace-nowrap text-[clamp(0.875rem,4vw,1.5rem)] font-semibold"
              :class="(active?.change ?? (latest - anchor)) >= 0 ? 'text-emerald-400' : 'text-rose-400'"
            >
              {{ formatSignedCurrency(active?.change ?? (latest - anchor)) }}
              ({{ formatSignedPercent(active?.changePct ?? (anchor ? (latest - anchor) / anchor : 0)) }})
            </p>
            <p class="numeric hidden text-xs text-slate-500 sm:block">
              {{ active?.open
                ? `Open (unrealized) · ${dayLabel(active.point.day)}`
                : active
                  ? `Realized · ${dayLabel(active.point.day)}`
                  : openPoint
                    ? `Open (unrealized) · ${dayLabel(openPoint.day)}`
                    : `Last realized · ${dayLabel(points[points.length - 1]?.day)}` }}
            </p>
          </div>
        </div>
      </div>

      <div
        ref="chartHost"
        class="min-w-0"
      >
        <svg
          :viewBox="`0 0 ${geometry.width} ${height}`"
          class="w-full touch-pan-y"
          :style="{ height: `${height}px` }"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          :aria-label="`Strategy equity from ${dayLabel(points[0]?.day)} to ${dayLabel(displayPoints[displayPoints.length - 1]?.day)}`"
          @pointerdown="onMove"
          @pointermove="onMove"
          @pointerleave="hovered = null"
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
              :x1="geometry.padding.left"
              :x2="geometry.width - geometry.padding.right"
              :y1="row.y"
              :y2="row.y"
              stroke="#1e293b"
              stroke-width="1"
            />
            <text
              v-for="row in grid"
              :key="`label-${row.value}`"
              :x="geometry.padding.left - 6"
              :y="row.y + 4"
              text-anchor="end"
              fill="#64748b"
              :font-size="compact ? 10 : 11"
            >{{ formatAxisCurrency(row.value, scale.max - scale.min) }}</text>
          </g>

          <line
            v-if="baselineY !== null"
            :x1="geometry.padding.left"
            :x2="geometry.width - geometry.padding.right"
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

          <path
            v-if="openSegment"
            :d="openSegment"
            fill="none"
            stroke="#fbbf24"
            stroke-width="2"
            stroke-dasharray="6 5"
            stroke-linecap="round"
            vector-effect="non-scaling-stroke"
          />
          <circle
            v-if="openPoint"
            :cx="scale.x(displayPoints.length - 1)"
            :cy="scale.y(openPoint.equity)"
            r="4"
            fill="#fbbf24"
            stroke="#020617"
            stroke-width="2"
          />

          <g v-if="active">
            <line
              :x1="active.x"
              :x2="active.x"
              :y1="geometry.padding.top"
              :y2="height - geometry.padding.bottom"
              stroke="#64748b"
              stroke-width="1"
              stroke-dasharray="3 3"
            />
            <circle
              :cx="active.x"
              :cy="active.y"
              r="4"
              :fill="active.open ? '#fbbf24' : gaining ? '#34d399' : '#fb7185'"
              stroke="#020617"
              stroke-width="2"
            />
          </g>

          <text
            v-for="tick in ticks"
            :key="`tick-${tick.index}`"
            :x="tick.x"
            :y="height - 6"
            :text-anchor="tick.anchor"
            fill="#64748b"
            :font-size="compact ? 10 : 11"
          >{{ tick.label }}</text>
        </svg>
      </div>
    </template>
  </div>
</template>

<style scoped>
.equity-chart {
  -webkit-user-select: none;
  -webkit-touch-callout: none;
  -webkit-tap-highlight-color: transparent;
}
</style>
