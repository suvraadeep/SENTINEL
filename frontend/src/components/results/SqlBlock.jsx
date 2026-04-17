import React, { useState } from 'react'
import { ChevronDown, ChevronRight, Copy, Check, BarChart2 } from 'lucide-react'
import clsx from 'clsx'

function formatSql(sql) {
  // Basic keyword highlighting via spans (no external lib needed)
  const keywords = /\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING|JOIN|LEFT|RIGHT|INNER|ON|AS|AND|OR|NOT|IN|LIKE|BETWEEN|LIMIT|OFFSET|WITH|UNION|ALL|DISTINCT|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|CREATE|TABLE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|DROP|ALTER|CAST|COALESCE|NULLIF|DATE|INTERVAL)\b/g
  return sql
}

export default function SqlBlock({ sql, resultPreview, aqpCi }) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showPreview, setShowPreview] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(sql).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const previewCols = resultPreview?.length > 0 ? Object.keys(resultPreview[0]) : []

  return (
    <div className="space-y-3">
      {/* SQL header */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 text-sm font-medium text-sentinel-muted hover:text-sentinel-text transition-colors"
        >
          {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          Generated SQL
        </button>
        <div className="flex items-center gap-2">
          {resultPreview?.length > 0 && (
            <button
              onClick={() => setShowPreview((v) => !v)}
              className="btn-ghost text-xs gap-1"
            >
              <BarChart2 className="w-3.5 h-3.5" />
              {showPreview ? 'Hide' : 'Show'} data ({resultPreview.length} rows)
            </button>
          )}
          <button onClick={handleCopy} className="btn-ghost text-xs gap-1">
            {copied ? <Check className="w-3.5 h-3.5 text-sentinel-green" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      {/* SQL code */}
      {expanded && (
        <div className="relative animate-fade-in">
          <pre className="sql-code overflow-x-auto max-h-64">{sql}</pre>
        </div>
      )}

      {/* AQP confidence intervals */}
      {aqpCi && Object.keys(aqpCi).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(aqpCi).slice(0, 3).map(([col, ci]) => (
            <div key={col} className="px-3 py-1.5 rounded-lg bg-sentinel-blue/5 border border-sentinel-blue/10 text-xs">
              <span className="text-sentinel-faint">{col}: </span>
              <span className="text-sentinel-blue font-medium">
                {ci.ci_lower?.toFixed(2)} ↔ {ci.ci_upper?.toFixed(2)}
              </span>
              <span className="text-sentinel-faint ml-1">(95% CI)</span>
            </div>
          ))}
        </div>
      )}

      {/* Data preview table */}
      {showPreview && resultPreview?.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-sentinel-border animate-fade-in">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-sentinel-border bg-sentinel-card">
                {previewCols.map((col) => (
                  <th key={col} className="px-3 py-2.5 text-left font-medium text-sentinel-faint uppercase tracking-wide whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {resultPreview.map((row, i) => (
                <tr key={i} className={clsx(
                  'border-b border-sentinel-border/50 transition-colors hover:bg-sentinel-card/50',
                  i % 2 === 0 ? '' : 'bg-sentinel-card/20'
                )}>
                  {previewCols.map((col) => (
                    <td key={col} className="px-3 py-2 text-sentinel-muted whitespace-nowrap">
                      {row[col] != null ? String(row[col]) : <span className="text-sentinel-faint italic">null</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
