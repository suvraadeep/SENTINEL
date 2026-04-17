import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle, AlertCircle, Info } from 'lucide-react'
import clsx from 'clsx'

const SEVERITY_CONFIG = {
  critical: { color: 'text-sentinel-red',    bg: 'bg-sentinel-red/10   border-sentinel-red/20',   icon: AlertCircle, label: 'Critical' },
  high:     { color: 'text-sentinel-yellow', bg: 'bg-sentinel-yellow/10 border-sentinel-yellow/20',icon: AlertTriangle,label: 'High' },
  medium:   { color: 'text-sentinel-blue',   bg: 'bg-sentinel-blue/10  border-sentinel-blue/20',  icon: Info,        label: 'Medium' },
  low:      { color: 'text-sentinel-faint',  bg: 'bg-sentinel-card     border-sentinel-border',   icon: Info,        label: 'Low' },
}

function getSeverity(anomaly) {
  // Prefer the backend-computed string label (CRITICAL / HIGH / MEDIUM / LOW)
  const label = (anomaly.severity || '').toLowerCase()
  if (label === 'critical') return 'critical'
  if (label === 'high')     return 'high'
  if (label === 'medium')   return 'medium'
  if (label === 'low')      return 'low'
  // Fallback: derive from z_score
  const score = anomaly.score || anomaly.z_score || 0
  const absScore = Math.abs(typeof score === 'number' ? score : parseFloat(score) || 0)
  if (absScore >= 4) return 'critical'
  if (absScore >= 3) return 'high'
  if (absScore >= 2) return 'medium'
  return 'low'
}

export default function AnomalyBlock({ data }) {
  if (!data || typeof data !== 'object') return null

  const anomalies = data.anomalies || data.outliers || data.detected || []
  const summary   = data.summary || data.narrative || data.alert || ''
  const stats = {
    total:    data.total_anomalies    ?? data.count ?? anomalies.length,
    critical: data.critical_count     ?? anomalies.filter((a) => getSeverity(a) === 'critical').length,
    high:     data.high_count         ?? anomalies.filter((a) => getSeverity(a) === 'high').length,
  }

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border text-center">
          <div className="text-2xl font-bold text-sentinel-text">{stats.total}</div>
          <div className="text-xs text-sentinel-faint mt-0.5">Total Anomalies</div>
        </div>
        <div className="p-3 rounded-xl bg-sentinel-red/10 border border-sentinel-red/20 text-center">
          <div className="text-2xl font-bold text-sentinel-red">{stats.critical}</div>
          <div className="text-xs text-sentinel-faint mt-0.5">Critical</div>
        </div>
        <div className="p-3 rounded-xl bg-sentinel-yellow/10 border border-sentinel-yellow/20 text-center">
          <div className="text-2xl font-bold text-sentinel-yellow">{stats.high}</div>
          <div className="text-xs text-sentinel-faint mt-0.5">High Severity</div>
        </div>
      </div>

      {/* Summary alert */}
      {summary && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-sentinel-yellow/5 border border-sentinel-yellow/20">
          <AlertTriangle className="w-5 h-5 text-sentinel-yellow flex-shrink-0 mt-0.5" />
          <div className="text-sm text-sentinel-muted leading-relaxed prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="text-sm text-sentinel-muted leading-relaxed mb-1 last:mb-0">{children}</p>,
                strong: ({ children }) => <strong className="font-semibold text-sentinel-text">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
              }}
            >
              {summary}
            </ReactMarkdown>
          </div>
        </div>
      )}

      {/* Anomaly list */}
      {anomalies.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider">
            Detected Anomalies
          </div>
          {anomalies.slice(0, 10).map((anomaly, i) => {
            const sev = getSeverity(anomaly)
            const cfg = SEVERITY_CONFIG[sev]
            const Icon = cfg.icon
            const name = anomaly.name || anomaly.metric || anomaly.column || anomaly.description || `Anomaly ${i + 1}`
            const score = anomaly.severity || anomaly.z_score || anomaly.score
            const date = anomaly.date || anomaly.timestamp || anomaly.time || null
            const value = anomaly.value || anomaly.actual || null
            return (
              <div key={i} className={clsx('flex items-start gap-3 p-3 rounded-xl border', cfg.bg)}>
                <Icon className={clsx('w-4 h-4 flex-shrink-0 mt-0.5', cfg.color)} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-sentinel-text">{name}</span>
                    <span className={clsx('badge text-xs', {
                      'badge-red': sev === 'critical',
                      'badge-yellow': sev === 'high',
                      'badge-blue': sev === 'medium',
                    })}>
                      {cfg.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-sentinel-faint flex-wrap">
                    {date && <span>{date}</span>}
                    {value != null && <span>Value: <strong className={cfg.color}>{typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}</strong></span>}
                    {score != null && <span>Z-score: <strong className={cfg.color}>{typeof score === 'number' ? score.toFixed(2) : score}</strong></span>}
                  </div>
                </div>
              </div>
            )
          })}
          {anomalies.length > 10 && (
            <div className="text-xs text-sentinel-faint text-center py-2">
              +{anomalies.length - 10} more anomalies
            </div>
          )}
        </div>
      )}

      {/* Raw fallback */}
      {!summary && anomalies.length === 0 && (
        <pre className="text-xs text-sentinel-muted bg-sentinel-card p-4 rounded-xl border border-sentinel-border whitespace-pre-wrap overflow-auto max-h-48">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
