import React, { useMemo, useState, useRef, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import MarkdownMessage from '../components/MarkdownMessage'
import clsx from 'clsx'
import {
  BarChart3, TrendingUp, Database, Clock, Zap, Target,
  ChevronDown, ChevronUp, ChevronRight,
  Bot, Sparkles, HelpCircle, Maximize2,
  FileText, Image, Code2, History, Layers,
} from 'lucide-react'

// ─────────────────────────────────────────────────────────────────────────────
// ChartCard — same pattern as Anomaly/RCA dashboards
// ─────────────────────────────────────────────────────────────────────────────
function ChartCard({ chart }) {
  const [explOpen, setExplOpen]   = useState(false)
  const [expanded, setExpanded]   = useState(false)
  const iframeRef = useRef(null)

  useEffect(() => {
    const iframe = iframeRef.current
    if (!iframe || !chart.html) return
    const doc = iframe.contentDocument || iframe.contentWindow?.document
    if (!doc) return
    doc.open()
    doc.write(`<!doctype html><html><head>
      <meta charset="utf-8">
      <style>* { box-sizing:border-box; } body { margin:0; background:transparent; overflow:hidden; }</style>
    </head><body>${chart.html}</body></html>`)
    doc.close()
  }, [chart.html, expanded])

  const expl = chart.explanation
  const hasExpl = expl && (expl.why || expl.what || expl.how_to_read)

  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden hover:border-sentinel-blue/25 transition-colors">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-sentinel-border bg-sentinel-surface/50">
        <span className="text-xs font-semibold text-sentinel-text truncate">{chart.title}</span>
        <div className="flex items-center gap-1 flex-shrink-0">
          {hasExpl && (
            <button
              onClick={() => setExplOpen(v => !v)}
              title="Why this chart?"
              className={clsx(
                'p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1',
                explOpen ? 'bg-sentinel-blue/20 text-sentinel-blue' : 'text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover'
              )}
            >
              <HelpCircle className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Why?</span>
            </button>
          )}
          <button
            onClick={() => setExpanded(v => !v)}
            title="Expand"
            className="p-1.5 rounded-lg text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover transition-colors"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      {explOpen && hasExpl && (
        <div className="px-4 py-3 bg-sentinel-blue/5 border-b border-sentinel-blue/20 space-y-2">
          {expl.why && (
            <div>
              <span className="text-[10px] font-bold text-sentinel-blue uppercase tracking-wider">Why this chart</span>
              <p className="text-xs text-sentinel-muted mt-0.5">{expl.why}</p>
            </div>
          )}
          {expl.what && (
            <div>
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">What it shows</span>
              <p className="text-xs text-sentinel-muted mt-0.5">{expl.what}</p>
            </div>
          )}
          {expl.how_to_read && (
            <div>
              <span className="text-[10px] font-bold text-amber-500 uppercase tracking-wider">How to read it</span>
              <p className="text-xs text-sentinel-muted mt-0.5">{expl.how_to_read}</p>
            </div>
          )}
        </div>
      )}
      <iframe
        ref={iframeRef}
        className="w-full"
        style={{ height: expanded ? 500 : 280, border: 'none', display: 'block' }}
        title={chart.title}
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// HistoryEntry — collapsible analysis result with charts + insights
// ─────────────────────────────────────────────────────────────────────────────
function HistoryEntry({ entry, index }) {
  const [open, setOpen] = useState(index === 0)

  const intentColors = {
    sql_query:  'bg-sentinel-blue/15 text-sentinel-blue border-sentinel-blue/30',
    rca:        'bg-purple-500/15 text-purple-400 border-purple-500/30',
    forecast:   'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    anomaly:    'bg-amber-500/15 text-amber-400 border-amber-500/30',
    math:       'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    analysis:   'bg-sentinel-blue/15 text-sentinel-blue border-sentinel-blue/30',
  }
  const ic = intentColors[entry.type] ?? intentColors.sql_query

  const hasContent = (entry.charts?.length > 0) || entry.insights || entry.sql

  return (
    <div className={clsx(
      'bg-sentinel-card border rounded-xl overflow-hidden transition-colors',
      open ? 'border-sentinel-blue/20' : 'border-sentinel-border hover:border-sentinel-border/80'
    )}>
      {/* Row */}
      <button
        onClick={() => hasContent && setOpen(v => !v)}
        className={clsx(
          'w-full flex items-center gap-3 px-4 py-3 text-left',
          hasContent ? 'cursor-pointer hover:bg-sentinel-hover/30' : 'cursor-default'
        )}
      >
        <div className={clsx('flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold', ic)}>
          {index + 1}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm text-sentinel-text truncate font-medium">{entry.query || 'Analysis'}</div>
          <div className="flex items-center gap-2 mt-0.5 flex-wrap">
            <span className="text-[10px] text-sentinel-faint">
              {new Date(entry.timestamp).toLocaleString()}
            </span>
            <span className={clsx('text-[10px] px-1.5 py-0.5 rounded-full border', ic)}>
              {entry.type === 'sql_query' ? 'SQL' : entry.type}
            </span>
            {entry.charts?.length > 0 && (
              <span className="text-[10px] text-sentinel-faint flex items-center gap-1">
                <Image className="w-2.5 h-2.5" />
                {entry.charts.length} chart{entry.charts.length !== 1 ? 's' : ''}
              </span>
            )}
            {entry.insights && (
              <span className="text-[10px] text-sentinel-faint flex items-center gap-1">
                <Sparkles className="w-2.5 h-2.5" />
                insights
              </span>
            )}
            {entry.sql && (
              <span className="text-[10px] text-sentinel-faint flex items-center gap-1">
                <Code2 className="w-2.5 h-2.5" />
                SQL
              </span>
            )}
          </div>
        </div>
        {hasContent && (
          open
            ? <ChevronUp className="w-4 h-4 text-sentinel-faint flex-shrink-0" />
            : <ChevronDown className="w-4 h-4 text-sentinel-faint flex-shrink-0" />
        )}
      </button>

      {/* Expanded content */}
      {open && hasContent && (
        <div className="border-t border-sentinel-border space-y-0">
          {/* AI Insights */}
          {entry.insights && (
            <div className="px-4 py-3 border-b border-sentinel-border/60">
              <div className="flex items-center gap-1.5 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-sentinel-blue" />
                <span className="text-xs font-semibold text-sentinel-text">AI Insights</span>
              </div>
              <div className="text-xs">
                <MarkdownMessage content={entry.insights} />
              </div>
            </div>
          )}

          {/* SQL */}
          {entry.sql && (
            <div className="px-4 py-3 border-b border-sentinel-border/60">
              <div className="flex items-center gap-1.5 mb-2">
                <Code2 className="w-3.5 h-3.5 text-sentinel-cyan" />
                <span className="text-xs font-semibold text-sentinel-text">SQL Query</span>
              </div>
              <pre className="text-xs text-sentinel-cyan font-mono bg-sentinel-surface rounded-lg px-3 py-2 overflow-x-auto whitespace-pre-wrap break-words">
                {entry.sql}
              </pre>
            </div>
          )}

          {/* Charts */}
          {entry.charts?.length > 0 && (
            <div className="px-4 py-3">
              <div className="flex items-center gap-1.5 mb-3">
                <Image className="w-3.5 h-3.5 text-sentinel-purple" />
                <span className="text-xs font-semibold text-sentinel-text">
                  Charts ({entry.charts.length})
                </span>
              </div>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                {entry.charts.map((chart, ci) => (
                  <ChartCard key={`${chart.title}-${ci}`} chart={chart} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DataSection — one pane for Original or Modified data
// ─────────────────────────────────────────────────────────────────────────────
function DataSection({ label, icon: Icon, color, entries, queryHistory, emptyHint }) {
  const [historyOpen, setHistoryOpen] = useState(true)
  const [perfOpen, setPerfOpen]       = useState(false)

  const stats = useMemo(() => {
    if (!queryHistory.length) return null
    let totalMs = 0, cacheHits = 0, totalCharts = 0, sqlCount = 0
    const intents = {}
    for (const q of queryHistory) {
      const intent = q.intent || 'sql_query'
      intents[intent] = (intents[intent] || 0) + 1
      totalMs    += q.duration_ms || 0
      if (q.memory_info?.cache_hit) cacheHits++
      totalCharts += (q.charts || []).length
      if (q.sql) sqlCount++
    }
    return { intents, totalMs, cacheHits, totalCharts, sqlCount }
  }, [queryHistory])

  const isEmpty = entries.length === 0

  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className={clsx('flex items-center gap-2 px-1')}>
        <div className={clsx('p-1.5 rounded-lg', color.bg)}>
          <Icon className={clsx('w-4 h-4', color.icon)} />
        </div>
        <div>
          <h2 className="text-sm font-bold text-sentinel-text">{label}</h2>
          <p className="text-[11px] text-sentinel-faint">
            {entries.length} analysis result{entries.length !== 1 ? 's' : ''}
            {queryHistory.length > 0 && ` · ${queryHistory.length} quer${queryHistory.length !== 1 ? 'ies' : 'y'}`}
          </p>
        </div>
      </div>

      {isEmpty ? (
        <div className="flex flex-col items-center justify-center py-10 gap-3 bg-sentinel-card border border-sentinel-border rounded-xl">
          <Icon className={clsx('w-8 h-8', color.icon, 'opacity-30')} />
          <div className="text-center">
            <div className="text-sm font-medium text-sentinel-muted">{emptyHint.title}</div>
            <div className="text-xs text-sentinel-faint mt-1">{emptyHint.sub}</div>
          </div>
        </div>
      ) : (
        <>
          {/* Analysis history accordion */}
          <div className="bg-sentinel-surface border border-sentinel-border rounded-xl overflow-hidden">
            <button
              onClick={() => setHistoryOpen(v => !v)}
              className="w-full flex items-center justify-between px-4 py-3 border-b border-sentinel-border hover:bg-sentinel-hover/30 transition-colors"
            >
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-sentinel-blue" />
                <span className="text-sm font-semibold text-sentinel-text">Analysis History</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-sentinel-blue/10 text-sentinel-blue border border-sentinel-blue/20">
                  {entries.length}
                </span>
              </div>
              {historyOpen
                ? <ChevronUp className="w-4 h-4 text-sentinel-faint" />
                : <ChevronDown className="w-4 h-4 text-sentinel-faint" />}
            </button>
            {historyOpen && (
              <div className="p-4 space-y-3">
                {entries.map((entry, i) => (
                  <HistoryEntry key={entry.id ?? i} entry={entry} index={i} />
                ))}
              </div>
            )}
          </div>

          {/* Performance metrics (collapsible) */}
          {queryHistory.length > 0 && (
            <div className="bg-sentinel-surface border border-sentinel-border rounded-xl overflow-hidden">
              <button
                onClick={() => setPerfOpen(v => !v)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-sentinel-hover/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-sentinel-purple" />
                  <span className="text-sm font-semibold text-sentinel-text">Query Performance</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-sentinel-purple/10 text-sentinel-purple border border-sentinel-purple/20">
                    {queryHistory.length} queries
                  </span>
                </div>
                {perfOpen
                  ? <ChevronUp className="w-4 h-4 text-sentinel-faint" />
                  : <ChevronDown className="w-4 h-4 text-sentinel-faint" />}
              </button>
              {perfOpen && (
                <div className="border-t border-sentinel-border p-4 space-y-4">
                  {/* Perf mini-cards */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {[
                      { label: 'Total Queries',   value: queryHistory.length,   icon: Database,   color: 'text-sentinel-blue' },
                      { label: 'Avg Response',    value: stats ? `${((stats.totalMs / queryHistory.length) / 1000).toFixed(1)}s` : '—', icon: Clock, color: 'text-sentinel-cyan' },
                      { label: 'Cache Hits',      value: stats ? `${Math.round((stats.cacheHits / queryHistory.length) * 100)}%` : '—', icon: Zap, color: 'text-sentinel-yellow' },
                      { label: 'Charts Made',     value: stats?.totalCharts ?? 0, icon: Image,    color: 'text-sentinel-purple' },
                    ].map(({ label, value, icon: Ic, color: c }) => (
                      <div key={label} className="p-3 rounded-lg bg-sentinel-card border border-sentinel-border">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[10px] text-sentinel-faint">{label}</span>
                          <Ic className={clsx('w-3.5 h-3.5', c)} />
                        </div>
                        <div className={clsx('text-lg font-bold', c)}>{value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Recent queries */}
                  <div>
                    <div className="text-[10px] font-semibold text-sentinel-faint uppercase tracking-wider mb-2">
                      Recent Queries
                    </div>
                    <div className="space-y-1.5">
                      {queryHistory.slice(0, 8).map((q, qi) => (
                        <div key={q.id || qi} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border">
                          <ChevronRight className="w-3 h-3 text-sentinel-faint flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="text-xs text-sentinel-text truncate">{q.query || 'Query'}</div>
                            <div className="text-[10px] text-sentinel-faint mt-0.5">
                              {q.timestamp ? new Date(q.timestamp).toLocaleTimeString() : ''}
                              {q.intent && ` · ${q.intent}`}
                              {q.memory_info?.cache_hit && ' · ⚡ cached'}
                            </div>
                          </div>
                          <span className="text-[10px] text-sentinel-faint flex-shrink-0">
                            {q.duration_ms ? `${(q.duration_ms / 1000).toFixed(1)}s` : '—'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AnalyticsPage — Insights Hub
// ─────────────────────────────────────────────────────────────────────────────
export default function AnalyticsPage() {
  const { analysisHistory, queryHistory, hasModified } = useApp()
  const [activeTab, setActiveTab] = useState('original')

  // Split analysis history and query history by data version
  const originalEntries = useMemo(
    () => analysisHistory.filter(e => e.version !== 'modified'),
    [analysisHistory]
  )
  const modifiedEntries = useMemo(
    () => analysisHistory.filter(e => e.version === 'modified'),
    [analysisHistory]
  )
  const originalQueries = useMemo(
    () => queryHistory.filter(q => q.version !== 'modified'),
    [queryHistory]
  )
  const modifiedQueries = useMemo(
    () => queryHistory.filter(q => q.version === 'modified'),
    [queryHistory]
  )

  const tabs = [
    {
      id:    'original',
      label: 'Original Data',
      icon:  Database,
      color: { bg: 'bg-indigo-500/10', icon: 'text-indigo-400', tab: 'border-indigo-500/50 bg-indigo-500/10 text-indigo-300' },
      entries: originalEntries,
      queries: originalQueries,
      emptyHint: {
        title: 'No original-data analysis yet',
        sub:   'Run queries or charts in the Intelligence tab to see them here.',
      },
    },
    {
      id:    'modified',
      label: 'Modified Data',
      icon:  Layers,
      color: { bg: 'bg-cyan-500/10', icon: 'text-cyan-400', tab: 'border-cyan-500/50 bg-cyan-500/10 text-cyan-300' },
      entries: modifiedEntries,
      queries: modifiedQueries,
      emptyHint: {
        title: 'No modified-data analysis yet',
        sub:   'Switch to "Modified" in the Intelligence toggle and run a query after making DataLab edits.',
      },
    },
  ]

  const activeTabCfg = tabs.find(t => t.id === activeTab) ?? tabs[0]

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 no-scrollbar">
      {/* Page header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold text-sentinel-text flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-sentinel-blue" />
            Insights Hub
          </h1>
          <p className="text-sm text-sentinel-muted mt-1">
            Generated charts, AI insights, SQL history — all in one place
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-sentinel-faint">
          <Sparkles className="w-3.5 h-3.5 text-sentinel-blue" />
          <span>{analysisHistory.length} total analyses · {queryHistory.length} queries</span>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-1.5 p-1 bg-sentinel-card border border-sentinel-border rounded-xl w-fit">
        {tabs.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          const count = tab.id === 'original' ? originalEntries.length : modifiedEntries.length
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              disabled={tab.id === 'modified' && !hasModified && modifiedEntries.length === 0}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150',
                isActive
                  ? tab.color.tab + ' border'
                  : tab.id === 'modified' && !hasModified && modifiedEntries.length === 0
                    ? 'text-sentinel-faint/40 cursor-not-allowed'
                    : 'text-sentinel-muted hover:bg-sentinel-hover hover:text-sentinel-text'
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
              {count > 0 && (
                <span className={clsx(
                  'px-1.5 py-0.5 rounded-full text-[10px] font-bold',
                  isActive ? 'bg-white/20' : 'bg-sentinel-hover'
                )}>
                  {count}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <DataSection
        key={activeTab}
        label={activeTabCfg.label}
        icon={activeTabCfg.icon}
        color={activeTabCfg.color}
        entries={activeTabCfg.entries}
        queryHistory={activeTabCfg.queries}
        emptyHint={activeTabCfg.emptyHint}
      />
    </div>
  )
}
