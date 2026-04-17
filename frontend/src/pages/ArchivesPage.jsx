import React, { useState } from 'react'
import { useApp } from '../context/AppContext'
import { Archive, Database, BarChart3, ChevronDown, ChevronUp, Clock, Search } from 'lucide-react'
import clsx from 'clsx'
import SqlBlock from '../components/results/SqlBlock'
import InsightBlock from '../components/results/InsightBlock'

const INTENT_LABELS = {
  sql_query: 'SQL', rca: 'Root Cause', forecast: 'Forecast',
  anomaly: 'Anomaly', math: 'Math',
}
const INTENT_COLORS = {
  sql_query: 'badge-blue', rca: 'badge-purple', forecast: 'badge-cyan',
  anomaly: 'badge-yellow', math: 'badge-green',
}

function ArchiveEntry({ entry, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const { query = '', intent, sql, insights, charts = [], duration_ms, timestamp, memory_info } = entry
  const hasCharts = charts.length > 0

  return (
    <div className="rounded-xl border border-sentinel-border overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start gap-3 p-4 bg-sentinel-card hover:bg-sentinel-surface transition-colors text-left"
      >
        <Database className="w-4 h-4 text-sentinel-faint flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-sentinel-text truncate pr-4">{query || 'Query'}</div>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            {intent && (
              <span className={clsx('badge text-xs', INTENT_COLORS[intent] || 'badge-blue')}>
                {INTENT_LABELS[intent] || intent}
              </span>
            )}
            {hasCharts && (
              <span className="flex items-center gap-1 text-xs text-sentinel-faint">
                <BarChart3 className="w-3 h-3" />
                {charts.length} chart{charts.length !== 1 ? 's' : ''}
              </span>
            )}
            {memory_info?.cache_hit && (
              <span className="text-xs text-sentinel-green">⚡ cached</span>
            )}
            {timestamp && (
              <span className="flex items-center gap-1 text-xs text-sentinel-faint">
                <Clock className="w-3 h-3" />
                {new Date(timestamp).toLocaleTimeString()}
              </span>
            )}
            {duration_ms > 0 && (
              <span className="text-xs text-sentinel-faint ml-auto">
                {(duration_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 text-sentinel-faint flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-sentinel-faint flex-shrink-0" />
        )}
      </button>

      {open && (
        <div className="border-t border-sentinel-border p-4 space-y-4">
          {sql && (
            <div>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">SQL Query</div>
              <SqlBlock sql={sql} resultPreview={entry.sql_result_preview} aqpCi={entry.aqp_ci} />
            </div>
          )}
          {insights && (
            <div>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">Analysis</div>
              <InsightBlock text={insights} />
            </div>
          )}
          {hasCharts && (
            <div className="text-xs text-sentinel-faint italic">
              {charts.length} visualisation{charts.length !== 1 ? 's' : ''} generated — view in Intelligence tab
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function ArchivesPage() {
  const { queryHistory, clearChat } = useApp()
  const [search, setSearch] = useState('')

  const filtered = search.trim()
    ? queryHistory.filter((q) =>
        (q.query || '').toLowerCase().includes(search.toLowerCase()) ||
        (q.sql || '').toLowerCase().includes(search.toLowerCase())
      )
    : queryHistory

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-sentinel-text">Query Archives</h1>
          <p className="text-sm text-sentinel-muted mt-1">
            {queryHistory.length} quer{queryHistory.length !== 1 ? 'ies' : 'y'} this session
          </p>
        </div>
        {queryHistory.length > 0 && (
          <button
            onClick={clearChat}
            className="text-xs text-sentinel-faint hover:text-sentinel-red transition-colors px-3 py-1.5 rounded-lg border border-sentinel-border hover:border-sentinel-red/30"
          >
            Clear History
          </button>
        )}
      </div>

      {queryHistory.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-16 h-16 rounded-2xl bg-sentinel-card border border-sentinel-border flex items-center justify-center">
            <Archive className="w-8 h-8 text-sentinel-faint" />
          </div>
          <div className="text-center">
            <div className="text-sm font-medium text-sentinel-muted">No queries archived yet</div>
            <div className="text-xs text-sentinel-faint mt-1">
              Your query history will appear here after running analyses
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sentinel-faint" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search queries or SQL..."
              className="w-full pl-9 pr-4 py-2.5 bg-sentinel-card border border-sentinel-border rounded-xl text-sm text-sentinel-text placeholder-sentinel-faint outline-none focus:border-sentinel-blue/50 transition-colors"
            />
          </div>

          {/* Archive entries */}
          <div className="space-y-2">
            {filtered.length === 0 ? (
              <div className="text-sm text-sentinel-faint text-center py-8">No matching queries</div>
            ) : (
              filtered.map((entry, i) => (
                <ArchiveEntry key={entry.id || i} entry={entry} defaultOpen={i === 0} />
              ))
            )}
          </div>
        </>
      )}
    </div>
  )
}
