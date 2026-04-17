import React from 'react'
import { GitBranch, ArrowRight, AlertTriangle } from 'lucide-react'

export default function RcaBlock({ data }) {
  if (!data || typeof data !== 'object') return null

  // Try to extract structured fields
  const drivers = data.drivers || data.root_causes || data.causes || []
  const summary = data.summary || data.explanation || data.narrative || ''
  const pValues = data.p_values || data.granger || {}
  const correlations = data.correlations || {}

  return (
    <div className="space-y-4">
      {/* Summary */}
      {summary && (
        <div className="p-4 rounded-xl bg-sentinel-purple/5 border border-sentinel-purple/20">
          <div className="flex items-center gap-2 mb-2">
            <GitBranch className="w-4 h-4 text-sentinel-purple" />
            <span className="text-xs font-semibold text-sentinel-purple uppercase tracking-wider">
              Causal Analysis
            </span>
          </div>
          <p className="text-sm text-sentinel-muted leading-relaxed">{summary}</p>
        </div>
      )}

      {/* Causal drivers */}
      {drivers.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">
            Identified Drivers
          </div>
          <div className="space-y-2">
            {drivers.map((driver, i) => {
              const name = typeof driver === 'string' ? driver : driver.name || driver.factor || String(driver)
              const contribution = driver.contribution || driver.pct || null
              const pValue = driver.p_value || null
              return (
                <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-sentinel-card border border-sentinel-border">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0
                    ${i === 0 ? 'bg-sentinel-red/10 text-sentinel-red' :
                      i === 1 ? 'bg-sentinel-yellow/10 text-sentinel-yellow' :
                                'bg-sentinel-blue/10 text-sentinel-blue'}`}>
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-sentinel-text truncate">{name}</div>
                    {pValue != null && (
                      <div className="text-xs text-sentinel-faint">
                        p-value: <span className={pValue < 0.05 ? 'text-sentinel-green' : 'text-sentinel-yellow'}>{typeof pValue === 'number' ? pValue.toFixed(4) : pValue}</span>
                      </div>
                    )}
                  </div>
                  {contribution != null && (
                    <div className="flex flex-col items-end flex-shrink-0">
                      <span className="text-sm font-bold text-sentinel-text">
                        {typeof contribution === 'number'
                          ? (contribution > 1 ? contribution.toFixed(1) + '%' : (contribution * 100).toFixed(1) + '%')
                          : contribution}
                      </span>
                      <span className="text-xs text-sentinel-faint">impact</span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* P-values table */}
      {Object.keys(pValues).length > 0 && (
        <div>
          <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">
            Granger Causality Tests
          </div>
          <div className="overflow-x-auto rounded-xl border border-sentinel-border">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-sentinel-border bg-sentinel-card">
                  <th className="px-3 py-2 text-left text-sentinel-faint font-medium">Variable</th>
                  <th className="px-3 py-2 text-right text-sentinel-faint font-medium">p-value</th>
                  <th className="px-3 py-2 text-right text-sentinel-faint font-medium">Significant</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(pValues).map(([var_name, p]) => {
                  const sig = typeof p === 'number' && p < 0.05
                  return (
                    <tr key={var_name} className="border-b border-sentinel-border/50">
                      <td className="px-3 py-2 text-sentinel-muted">{var_name}</td>
                      <td className="px-3 py-2 text-right font-mono">{typeof p === 'number' ? p.toFixed(4) : p}</td>
                      <td className="px-3 py-2 text-right">
                        <span className={sig ? 'badge-green badge text-xs' : 'badge badge-red text-xs'}>
                          {sig ? 'Yes' : 'No'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Raw data fallback */}
      {!summary && drivers.length === 0 && Object.keys(pValues).length === 0 && (
        <div className="p-4 rounded-xl bg-sentinel-card border border-sentinel-border">
          <pre className="text-xs text-sentinel-muted whitespace-pre-wrap overflow-auto max-h-48">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
