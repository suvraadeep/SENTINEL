import React from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function ForecastBlock({ data }) {
  if (!data || typeof data !== 'object') return null

  const summary = data.summary || data.narrative || data.explanation || ''
  const horizon = data.horizon || data.forecast_days || 7
  const confidence = data.confidence_interval || data.ci || '90%'
  const trend = data.trend || data.direction || null
  const changepoints = data.changepoints || []
  const metrics = data.metrics || {}

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus
  const trendColor = trend === 'up' ? 'text-sentinel-green' : trend === 'down' ? 'text-sentinel-red' : 'text-sentinel-muted'

  return (
    <div className="space-y-4">
      {/* Key metrics row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border text-center">
          <div className="text-xl font-bold text-sentinel-cyan">{horizon}d</div>
          <div className="text-xs text-sentinel-faint mt-0.5">Horizon</div>
        </div>
        <div className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border text-center">
          <div className={`text-xl font-bold ${trendColor} flex items-center justify-center gap-1`}>
            <TrendIcon className="w-5 h-5" />
            <span className="capitalize">{trend || 'Neutral'}</span>
          </div>
          <div className="text-xs text-sentinel-faint mt-0.5">Trend</div>
        </div>
        <div className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border text-center">
          <div className="text-xl font-bold text-sentinel-purple">
            {typeof confidence === 'number' ? `${(confidence * 100).toFixed(0)}%` : confidence}
          </div>
          <div className="text-xs text-sentinel-faint mt-0.5">Confidence</div>
        </div>
      </div>

      {/* Summary */}
      {summary && (
        <div className="p-4 rounded-xl bg-sentinel-cyan/5 border border-sentinel-cyan/20">
          <div className="text-xs font-semibold text-sentinel-cyan uppercase tracking-wider mb-2">Forecast Summary</div>
          <p className="text-sm text-sentinel-muted leading-relaxed">{summary}</p>
        </div>
      )}

      {/* Changepoints */}
      {changepoints.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">
            Detected Changepoints
          </div>
          <div className="space-y-1.5">
            {changepoints.slice(0, 5).map((cp, i) => (
              <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border">
                <div className="w-2 h-2 rounded-full bg-sentinel-yellow flex-shrink-0" />
                <span className="text-sm text-sentinel-muted">{typeof cp === 'string' ? cp : JSON.stringify(cp)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional metrics */}
      {Object.keys(metrics).length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} className="px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border">
              <div className="text-xs text-sentinel-faint capitalize">{key.replace(/_/g, ' ')}</div>
              <div className="text-sm font-medium text-sentinel-text mt-0.5">
                {typeof val === 'number' ? val.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(val)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Raw fallback */}
      {!summary && changepoints.length === 0 && Object.keys(metrics).length === 0 && (
        <pre className="text-xs text-sentinel-muted bg-sentinel-card p-4 rounded-xl border border-sentinel-border whitespace-pre-wrap overflow-auto max-h-48">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
