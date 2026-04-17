import React, { useMemo } from 'react'
import { useApp } from '../context/AppContext'
import { AlertTriangle, AlertCircle, Info, Radio, TrendingDown, TrendingUp, Activity } from 'lucide-react'
import clsx from 'clsx'

function AnomalyCard({ anomaly, i }) {
  const score = Math.abs(anomaly.severity || anomaly.z_score || anomaly.score || 0)
  const sev = score >= 4 ? 'critical' : score >= 3 ? 'high' : score >= 2 ? 'medium' : 'low'
  const cfg = {
    critical: { color: 'text-sentinel-red',    bg: 'bg-sentinel-red/10 border-sentinel-red/20',   Icon: AlertCircle, label: 'Critical' },
    high:     { color: 'text-sentinel-yellow', bg: 'bg-sentinel-yellow/10 border-sentinel-yellow/20', Icon: AlertTriangle, label: 'High' },
    medium:   { color: 'text-sentinel-blue',   bg: 'bg-sentinel-blue/10 border-sentinel-blue/20',  Icon: Info, label: 'Medium' },
    low:      { color: 'text-sentinel-faint',  bg: 'bg-sentinel-card border-sentinel-border',       Icon: Info, label: 'Low' },
  }[sev]
  const { Icon } = cfg
  const name = anomaly.name || anomaly.metric || anomaly.column || anomaly.description || `Signal ${i + 1}`
  return (
    <div className={clsx('flex items-start gap-3 p-3 rounded-xl border', cfg.bg)}>
      <Icon className={clsx('w-4 h-4 flex-shrink-0 mt-0.5', cfg.color)} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-medium text-sentinel-text">{name}</span>
          <span className={clsx('text-xs px-1.5 py-0.5 rounded font-medium border', cfg.bg, cfg.color)}>
            {cfg.label}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-1 text-xs text-sentinel-faint flex-wrap">
          {anomaly.date && <span>{anomaly.date}</span>}
          {anomaly.value != null && (
            <span>Value: <strong className={cfg.color}>
              {typeof anomaly.value === 'number' ? anomaly.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : anomaly.value}
            </strong></span>
          )}
          {score > 0 && <span>Z-score: <strong className={cfg.color}>{score.toFixed(2)}</strong></span>}
        </div>
      </div>
    </div>
  )
}

function ForecastPreview({ fc }) {
  const trend = fc.trend || fc.direction || ''
  const isUp = trend === 'up' || trend === 'increasing' || trend === 'positive'
  const isDown = trend === 'down' || trend === 'decreasing' || trend === 'negative'
  return (
    <div className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border">
      <div className="flex items-center gap-2 mb-2">
        {isUp ? (
          <TrendingUp className="w-4 h-4 text-sentinel-green" />
        ) : isDown ? (
          <TrendingDown className="w-4 h-4 text-sentinel-red" />
        ) : (
          <Activity className="w-4 h-4 text-sentinel-blue" />
        )}
        <span className="text-sm font-medium text-sentinel-text">{fc.query || 'Forecast Signal'}</span>
      </div>
      {fc.summary && <p className="text-xs text-sentinel-muted leading-relaxed">{fc.summary}</p>}
      {fc.horizon && (
        <div className="mt-2 text-xs text-sentinel-faint">Horizon: {fc.horizon}</div>
      )}
    </div>
  )
}

export default function SignalsPage() {
  const { queryHistory, memoryStats } = useApp()

  // Extract anomalies and forecasts from query history
  const { anomalies, forecasts, anomalyQueries } = useMemo(() => {
    const aList = []
    const fList = []
    const aqList = []
    for (const q of queryHistory) {
      if (q.anomaly_result) {
        const detected = q.anomaly_result.anomalies || q.anomaly_result.outliers || q.anomaly_result.detected || []
        if (detected.length > 0) {
          aList.push(...detected.map((a) => ({ ...a, _query: q.query })))
          aqList.push(q)
        } else if (q.anomaly_result.summary || q.anomaly_result.narrative) {
          aqList.push(q)
        }
      }
      if (q.forecast_result) {
        fList.push({ ...q.forecast_result, query: q.query })
      }
    }
    return { anomalies: aList, forecasts: fList, anomalyQueries: aqList }
  }, [queryHistory])

  const critCount = anomalies.filter((a) => {
    const s = Math.abs(a.severity || a.z_score || a.score || 0)
    return s >= 4
  }).length

  const highCount = anomalies.filter((a) => {
    const s = Math.abs(a.severity || a.z_score || a.score || 0)
    return s >= 3 && s < 4
  }).length

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-sentinel-text">Signals & Monitoring</h1>
          <p className="text-sm text-sentinel-muted mt-1">Anomalies and forecast signals from your analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="glow-dot bg-sentinel-green" />
          <span className="text-xs text-sentinel-faint">Live</span>
        </div>
      </div>

      {/* Signal summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-sentinel-red/10 border border-sentinel-red/20 text-center">
          <div className="text-2xl font-bold text-sentinel-red">{critCount}</div>
          <div className="text-xs text-sentinel-faint mt-1">Critical Signals</div>
        </div>
        <div className="p-4 rounded-xl bg-sentinel-yellow/10 border border-sentinel-yellow/20 text-center">
          <div className="text-2xl font-bold text-sentinel-yellow">{highCount}</div>
          <div className="text-xs text-sentinel-faint mt-1">High Severity</div>
        </div>
        <div className="p-4 rounded-xl bg-sentinel-blue/10 border border-sentinel-blue/20 text-center">
          <div className="text-2xl font-bold text-sentinel-blue">{forecasts.length}</div>
          <div className="text-xs text-sentinel-faint mt-1">Forecasts Active</div>
        </div>
      </div>

      {anomalies.length === 0 && forecasts.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-16 h-16 rounded-2xl bg-sentinel-card border border-sentinel-border flex items-center justify-center">
            <Radio className="w-8 h-8 text-sentinel-faint" />
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-sentinel-muted">No signals detected yet</div>
            <div className="text-xs text-sentinel-faint mt-1">
              Run anomaly detection or forecast queries in Intelligence
            </div>
            <div className="mt-3 text-xs text-sentinel-faint italic">
              Try: "Detect anomalies in revenue" or "Forecast next 7 days sales"
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Detected anomalies */}
          {anomalies.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-3">
                Detected Anomalies ({anomalies.length})
              </div>
              <div className="space-y-2">
                {anomalies.slice(0, 20).map((a, i) => (
                  <AnomalyCard key={i} anomaly={a} i={i} />
                ))}
              </div>
            </div>
          )}

          {/* Anomaly query summaries */}
          {anomalyQueries.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-3">
                Anomaly Analysis Results
              </div>
              <div className="space-y-2">
                {anomalyQueries.map((q, i) => (
                  <div key={i} className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border">
                    <div className="text-xs font-medium text-sentinel-blue mb-1">{q.query}</div>
                    {q.anomaly_result?.summary && (
                      <p className="text-xs text-sentinel-muted leading-relaxed">{q.anomaly_result.summary}</p>
                    )}
                    {q.anomaly_result?.narrative && (
                      <p className="text-xs text-sentinel-muted leading-relaxed">{q.anomaly_result.narrative}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Forecasts */}
          {forecasts.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-3">
                Active Forecasts ({forecasts.length})
              </div>
              <div className="space-y-2">
                {forecasts.map((fc, i) => (
                  <ForecastPreview key={i} fc={fc} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
