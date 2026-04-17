import React, { useState } from 'react'
import clsx from 'clsx'
import SqlBlock from '../results/SqlBlock'
import ChartFrame from '../results/ChartFrame'
import InsightBlock from '../results/InsightBlock'
import RcaBlock from '../results/RcaBlock'
import ForecastBlock from '../results/ForecastBlock'
import AnomalyBlock from '../results/AnomalyBlock'
import {
  Database, BarChart3, Lightbulb, GitBranch, TrendingUp,
  AlertTriangle, ChevronDown, ChevronUp, Brain, PieChart,
} from 'lucide-react'

// Collapsible section wrapper
function Section({ title, icon: Icon, color = 'text-sentinel-blue', children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-xl border border-sentinel-border overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 bg-sentinel-card hover:bg-sentinel-surface transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className={clsx('w-4 h-4', color)} />
          <span className="text-sm font-semibold text-sentinel-text">{title}</span>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 text-sentinel-faint" />
        ) : (
          <ChevronDown className="w-4 h-4 text-sentinel-faint" />
        )}
      </button>
      {open && (
        <div className="p-4 border-t border-sentinel-border">
          {children}
        </div>
      )}
    </div>
  )
}

export default function ResultCard({ result }) {
  const {
    sql, charts = [], insights, chart_explanations, error,
    rca_result, forecast_result, anomaly_result, math_result,
    sql_result_preview, aqp_ci, intent, memory_info, duration_ms,
  } = result

  // ── Chat / greeting / irrelevant — render as plain message, no badges ──
  if (intent === 'chat' || intent === 'greeting' || intent === 'irrelevant') {
    return (
      <div className="w-full">
        <InsightBlock text={insights || ''} />
      </div>
    )
  }

  const hasSql               = !!sql
  const hasCharts            = charts.length > 0
  const hasChartExplanations = !!chart_explanations
  const hasRca               = !!rca_result && Object.keys(rca_result).length > 0
  const hasForecast          = !!forecast_result && Object.keys(forecast_result).length > 0
  const hasAnomaly           = !!anomaly_result && Object.keys(anomaly_result).length > 0
  const hasMath              = !!math_result && Object.keys(math_result).length > 0

  return (
    <div className="w-full space-y-3">
      {/* Header: intent + badges */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          {intent && intent !== 'chat' && (
            <span className={clsx('badge text-xs', {
              'badge-blue':   intent === 'sql_query',
              'badge-purple': intent === 'rca' || intent === 'prediction',
              'badge-cyan':   intent === 'forecast',
              'badge-yellow': intent === 'anomaly',
              'badge-green':  intent === 'math',
            })}>
              {intent === 'sql_query'   ? 'SQL Query' :
               intent === 'rca'         ? 'Root Cause Analysis' :
               intent === 'forecast'    ? 'Forecast' :
               intent === 'anomaly'     ? 'Anomaly Detection' :
               intent === 'math'        ? 'Math Analysis' :
               intent === 'prediction'  ? 'ML Prediction' : intent}
            </span>
          )}
          {memory_info?.cache_hit && (
            <span className="badge badge-green text-xs">⚡ Cache Hit</span>
          )}
        </div>
        {duration_ms > 0 && (
          <span className="text-xs text-sentinel-faint">
            {(duration_ms / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      {/* ── 1. SQL Query ─────────────────────────────────────── */}
      {hasSql && (
        <Section title="SQL Query" icon={Database} color="text-sentinel-blue">
          <SqlBlock
            sql={sql}
            resultPreview={sql_result_preview}
            aqpCi={aqp_ci}
          />
        </Section>
      )}

      {/* ── 2. Charts / Visualisations ───────────────────────── */}
      {hasCharts && (
        <Section title={`Visualisations (${charts.length})`} icon={BarChart3} color="text-sentinel-cyan">
          <div className="space-y-4">
            {charts.map((chart, i) => (
              <ChartFrame key={i} chart={chart} />
            ))}
          </div>
        </Section>
      )}

      {/* ── 3. Chart-by-Chart Analysis ───────────────────────── */}
      {hasChartExplanations && (
        <Section title="Chart-by-Chart Analysis" icon={PieChart} color="text-sentinel-cyan" defaultOpen={false}>
          <InsightBlock text={chart_explanations} />
        </Section>
      )}

      {/* ── 4. Analysis & Insights ───────────────────────────── */}
      {insights && (
        <Section title="Analysis & Insights" icon={Lightbulb} color="text-sentinel-yellow" defaultOpen={true}>
          <InsightBlock text={insights} error={error} />
        </Section>
      )}

      {/* ── 5. Root Cause Analysis ────────────────────────────── */}
      {hasRca && (
        <Section title="Root Cause Analysis" icon={GitBranch} color="text-sentinel-purple">
          <RcaBlock data={rca_result} />
        </Section>
      )}

      {/* ── 6. Forecast ───────────────────────────────────────── */}
      {hasForecast && (
        <Section title="Forecast" icon={TrendingUp} color="text-sentinel-green">
          <ForecastBlock data={forecast_result} />
        </Section>
      )}

      {/* ── 7. Anomaly Alerts ─────────────────────────────────── */}
      {hasAnomaly && (
        <Section title="Anomaly Detection" icon={AlertTriangle} color="text-sentinel-yellow">
          <AnomalyBlock data={anomaly_result} />
        </Section>
      )}

      {/* ── 8. Math Result ────────────────────────────────────── */}
      {hasMath && (
        <Section title="Mathematical Analysis" icon={Brain} color="text-sentinel-purple">
          <pre className="text-xs text-sentinel-muted whitespace-pre-wrap leading-relaxed">
            {typeof math_result === 'string'
              ? math_result
              : JSON.stringify(math_result, null, 2)}
          </pre>
        </Section>
      )}

      {/* ── Error banner — shown whenever error is set ───────── */}
      {error && (
        <div className="flex items-start gap-3 p-3 rounded-xl bg-sentinel-red/5 border border-sentinel-red/20">
          <AlertTriangle className="w-4 h-4 text-sentinel-red flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-xs font-semibold text-sentinel-red mb-0.5">Pipeline Warning</div>
            <div className="text-xs text-sentinel-muted leading-relaxed">{error}</div>
          </div>
        </div>
      )}
    </div>
  )
}
