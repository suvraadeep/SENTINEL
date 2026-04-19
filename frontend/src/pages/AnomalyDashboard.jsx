import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import clsx from 'clsx'
import {
  AlertTriangle, Play, Loader2, MessageSquare, X,
  Send, ChevronDown, ChevronUp, ShieldAlert, ShieldCheck,
  TrendingDown, Activity, RefreshCw, Layers, Download,
  HelpCircle, Maximize2, Bot, Sparkles,
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import { detectAnomalies, anomalyChat } from '../api/client'
import MarkdownMessage from '../components/MarkdownMessage'

// ─────────────────────────────────────────────────────────────────────────────
// SeverityBadge
// ─────────────────────────────────────────────────────────────────────────────
function SeverityBadge({ severity }) {
  const cfg = {
    CRITICAL: 'bg-red-500/15 text-red-400 border-red-500/30',
    HIGH:     'bg-amber-500/15 text-amber-400 border-amber-500/30',
    MEDIUM:   'bg-blue-500/15 text-blue-400 border-blue-500/30',
  }
  return (
    <span className={clsx(
      'px-2 py-0.5 rounded-full text-[10px] font-semibold border',
      cfg[severity] ?? 'bg-sentinel-hover text-sentinel-muted border-sentinel-border'
    )}>
      {severity}
    </span>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// StatCard
// ─────────────────────────────────────────────────────────────────────────────
function StatCard({ label, value, icon: Icon, colorClass, delta }) {
  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-4 flex items-center gap-4 hover:border-sentinel-blue/20 transition-colors">
      <div className={clsx('p-2.5 rounded-xl', colorClass)}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-2xl font-bold text-sentinel-text tabular-nums">{(value ?? 0).toLocaleString()}</div>
        <div className="text-xs text-sentinel-faint truncate">{label}</div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MethodToggle — pill bar switching between detection method views
// ─────────────────────────────────────────────────────────────────────────────
function MethodToggle({ activeMethod, setActiveMethod, methods }) {
  const LABELS = {
    statistical: 'Statistical',
    ml:          'ML',
    timeseries:  'Time-Series',
    ensemble:    'Ensemble',
  }
  const COLORS = {
    statistical: 'border-violet-500/50 bg-violet-500/10 text-violet-300',
    ml:          'border-emerald-500/50 bg-emerald-500/10 text-emerald-300',
    timeseries:  'border-amber-500/50 bg-amber-500/10 text-amber-300',
    ensemble:    'border-blue-500/50 bg-blue-500/10 text-blue-300',
  }

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {Object.entries(LABELS).map(([key, label]) => {
        const stats  = methods?.[key]?.stats
        const count  = stats?.total ?? 0
        const active = activeMethod === key
        return (
          <button
            key={key}
            onClick={() => setActiveMethod(key)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all duration-150',
              active
                ? COLORS[key]
                : 'border-sentinel-border bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted hover:border-sentinel-muted/30'
            )}
          >
            {label}
            {count > 0 && (
              <span className={clsx(
                'px-1.5 py-0.5 rounded-full text-[10px] font-bold',
                active ? 'bg-white/20' : 'bg-sentinel-hover'
              )}>
                {count}
              </span>
            )}
            {key === 'ensemble' && <Sparkles className="w-3 h-3 opacity-70" />}
          </button>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ChartCard — wraps chart with collapsible "Why this chart?" explanation
// ─────────────────────────────────────────────────────────────────────────────
function ChartCard({ chart, agentGenerated }) {
  const [explOpen, setExplOpen] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const iframeRef = useRef(null)

  useEffect(() => {
    const iframe = iframeRef.current
    if (!iframe || !chart.html) return
    const doc = iframe.contentDocument || iframe.contentWindow?.document
    if (!doc) return
    doc.open()
    doc.write(`<!doctype html><html><head>
      <meta charset="utf-8">
      <style>
        * { box-sizing: border-box; }
        body { margin:0; background:transparent; overflow:hidden; }
        .plotly-graph-div { width:100% !important; }
      </style>
    </head><body>${chart.html}</body></html>`)
    doc.close()
  }, [chart.html, expanded])

  const expl = chart.explanation
  const hasExpl = expl && (expl.why || expl.what || expl.how_to_read)

  return (
    <div className={clsx(
      'bg-sentinel-card border rounded-xl overflow-hidden transition-all duration-200 group',
      agentGenerated ? 'border-violet-500/40' : 'border-sentinel-border hover:border-sentinel-blue/25'
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-sentinel-border bg-sentinel-surface/50">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-semibold text-sentinel-text truncate">{chart.title}</span>
          {agentGenerated && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-[10px] font-medium flex-shrink-0">
              <Bot className="w-2.5 h-2.5" />
              Added by AI
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {hasExpl && (
            <button
              onClick={() => setExplOpen(v => !v)}
              title="Why this chart?"
              className={clsx(
                'p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1',
                explOpen
                  ? 'bg-sentinel-blue/20 text-sentinel-blue'
                  : 'text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover'
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

      {/* Explanation panel */}
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

      {/* Chart iframe */}
      <iframe
        ref={iframeRef}
        className="w-full"
        style={{ height: expanded ? 500 : 300, border: 'none', display: 'block' }}
        title={chart.title}
        sandbox="allow-scripts allow-same-origin"
      />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyTable
// ─────────────────────────────────────────────────────────────────────────────
function AnomalyTable({ anomalies, table }) {
  const [sortKey, setSortKey]   = useState('ensemble_score')
  const [sortDir, setSortDir]   = useState('desc')
  const [filter, setFilter]     = useState('')
  const [selected, setSelected] = useState(null)   // row detail

  const sorted = useMemo(() => [...anomalies]
    .filter(a => !filter || JSON.stringify(a).toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      if (typeof av === 'number') return sortDir === 'desc' ? bv - av : av - bv
      return sortDir === 'desc'
        ? String(bv).localeCompare(String(av))
        : String(av).localeCompare(String(bv))
    }), [anomalies, filter, sortKey, sortDir])

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return null
    return sortDir === 'desc'
      ? <ChevronDown className="w-3 h-3 inline ml-0.5" />
      : <ChevronUp className="w-3 h-3 inline ml-0.5" />
  }

  const exportCsv = () => {
    const cols = ['row_index', 'column', 'value', 'baseline', 'z_score', 'ensemble_score', 'severity', 'date']
      .filter(c => sorted[0] && c in sorted[0])
    const header = cols.join(',')
    const rows   = sorted.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(','))
    const blob   = new Blob([header + '\n' + rows.join('\n')], { type: 'text/csv' })
    const url    = URL.createObjectURL(blob)
    const a      = document.createElement('a')
    a.href = url
    a.download = `anomalies_${table || 'export'}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const COLS = ['row_index', 'column', 'value', 'baseline', 'z_score', 'ensemble_score', 'severity']
  const hasDate = sorted[0]?.date != null

  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden">
      {/* Table header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sentinel-border gap-3 flex-wrap">
        <span className="text-sm font-semibold text-sentinel-text">
          Anomaly Records
          <span className="ml-2 text-xs font-normal text-sentinel-faint">({sorted.length.toLocaleString()})</span>
        </span>
        <div className="flex items-center gap-2 flex-wrap">
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter records..."
            className="bg-sentinel-surface border border-sentinel-border rounded-lg px-3 py-1.5 text-xs text-sentinel-text placeholder-sentinel-faint outline-none focus:border-sentinel-blue/50 w-44"
          />
          <button
            onClick={exportCsv}
            disabled={sorted.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sentinel-card border border-sentinel-border text-xs text-sentinel-muted hover:text-sentinel-text hover:border-sentinel-blue/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-sentinel-border bg-sentinel-surface/40">
              {[...COLS, ...(hasDate ? ['date'] : [])].map(col => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className="px-3 py-2.5 text-left text-sentinel-faint font-medium cursor-pointer hover:text-sentinel-muted select-none whitespace-nowrap"
                >
                  {col.replace(/_/g, ' ')}<SortIcon col={col} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 150).map((row, i) => (
              <tr
                key={i}
                onClick={() => setSelected(selected?.row_index === row.row_index && selected?.column === row.column ? null : row)}
                className={clsx(
                  'border-b border-sentinel-border/40 cursor-pointer transition-colors',
                  selected?.row_index === row.row_index && selected?.column === row.column
                    ? 'bg-sentinel-blue/10'
                    : row.severity === 'CRITICAL' ? 'bg-red-500/5 hover:bg-red-500/10'
                    : row.severity === 'HIGH'     ? 'bg-amber-500/5 hover:bg-amber-500/10'
                    : 'hover:bg-sentinel-hover/40'
                )}
              >
                <td className="px-3 py-2 text-sentinel-muted tabular-nums">{row.row_index}</td>
                <td className="px-3 py-2 text-sentinel-text font-mono">{row.column}</td>
                <td className="px-3 py-2 text-sentinel-text tabular-nums">
                  {typeof row.value === 'number' ? row.value.toFixed(4) : row.value}
                </td>
                <td className="px-3 py-2 text-sentinel-muted tabular-nums">
                  {typeof row.baseline === 'number' ? row.baseline.toFixed(4) : (row.baseline ?? '—')}
                </td>
                <td className="px-3 py-2 font-mono tabular-nums" style={{
                  color: Math.abs(row.z_score ?? 0) > 4 ? '#EF4444'
                       : Math.abs(row.z_score ?? 0) > 3 ? '#F59E0B' : '#94A3B8',
                }}>
                  {row.z_score != null ? row.z_score.toFixed(2) : '—'}
                </td>
                <td className="px-3 py-2">
                  {row.ensemble_score != null ? (
                    <div className="flex items-center gap-2">
                      <div className="w-14 h-1.5 bg-sentinel-hover rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.round((row.ensemble_score ?? 0) * 100)}%`,
                            background: (row.ensemble_score ?? 0) > 0.75 ? '#EF4444'
                                      : (row.ensemble_score ?? 0) > 0.5  ? '#F59E0B' : '#3B82F6',
                          }}
                        />
                      </div>
                      <span className="text-sentinel-muted tabular-nums">
                        {((row.ensemble_score ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ) : '—'}
                </td>
                <td className="px-3 py-2"><SeverityBadge severity={row.severity} /></td>
                {hasDate && (
                  <td className="px-3 py-2 text-sentinel-faint font-mono text-[10px]">
                    {String(row.date).slice(0, 19)}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>

        {sorted.length > 150 && (
          <div className="px-4 py-2 text-xs text-sentinel-faint border-t border-sentinel-border text-center">
            Showing 150 of {sorted.length.toLocaleString()} records — export CSV for full data
          </div>
        )}
        {sorted.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-sentinel-faint">
            No anomalies match the filter.
          </div>
        )}
      </div>

      {/* Row detail panel */}
      {selected && (
        <div className="border-t border-sentinel-border bg-sentinel-surface/60 px-4 py-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-sentinel-blue">
              Row {selected.row_index} — {selected.column}
            </span>
            <button
              onClick={() => setSelected(null)}
              className="p-1 rounded hover:bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {[
              { label: 'Observed', value: typeof selected.value === 'number' ? selected.value.toFixed(6) : selected.value },
              { label: 'Baseline', value: typeof selected.baseline === 'number' ? selected.baseline.toFixed(6) : (selected.baseline ?? '—') },
              { label: 'Z-score', value: selected.z_score != null ? selected.z_score.toFixed(3) : '—' },
              { label: 'Score', value: selected.ensemble_score != null ? `${(selected.ensemble_score * 100).toFixed(1)}%` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-sentinel-card border border-sentinel-border rounded-lg p-2.5">
                <div className="text-[10px] text-sentinel-faint mb-0.5">{label}</div>
                <div className="font-mono text-sentinel-text font-semibold">{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VersionToggle
// ─────────────────────────────────────────────────────────────────────────────
function VersionToggle({ version, setVersion, hasModified }) {
  return (
    <div className="flex items-center gap-2">
      <Layers className="w-3.5 h-3.5 text-sentinel-cyan" />
      <div className={clsx(
        'flex rounded-lg overflow-hidden border-2 transition-all duration-300',
        version === 'modified' && hasModified
          ? 'border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
          : 'border-indigo-500/40 shadow-[0_0_8px_rgba(99,102,241,0.1)]',
      )}>
        {['original', 'modified'].map(v => (
          <button
            key={v}
            onClick={() => hasModified && setVersion(v)}
            disabled={v === 'modified' && !hasModified}
            className={clsx(
              'px-3.5 py-1.5 text-xs font-bold transition-all duration-200 capitalize',
              version === v
                ? v === 'modified' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-indigo-500/20 text-indigo-300'
                : v === 'modified' && !hasModified
                  ? 'bg-sentinel-card text-sentinel-faint/40 cursor-not-allowed'
                  : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted cursor-pointer',
            )}
          >
            {v === 'original' ? '⬡ Original' : '⬢ Modified'}
          </button>
        ))}
      </div>
      {!hasModified && (
        <span className="text-[9px] text-sentinel-faint/50 italic">No modified version</span>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ChatPanel — Anomaly AI chat with MarkdownMessage + agent chart injection
// ─────────────────────────────────────────────────────────────────────────────
function ChatPanel({ onClose, context, table, onPendingCharts, messages, setMessages }) {
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [toolBadges, setToolBadges] = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async () => {
    const msg = input.trim()
    if (!msg || loading) return
    setInput('')
    const userMsg = { role: 'user', text: msg }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setToolBadges([])

    // Build chat history (last 10, excluding welcome message)
    const history = messages
      .filter(m => m.role !== 'system')
      .slice(-10)
      .map(m => ({ role: m.role, content: m.text }))

    try {
      const res = await anomalyChat({
        message:      msg,
        context,
        table:        table || context?.table,
        chat_history: history,
      })

      // Send agent charts to approval panel instead of injecting directly
      if (res.charts?.length > 0) {
        onPendingCharts?.({ charts: res.charts, note: res.response?.slice(0, 120) ?? '' })
      }
      if (res.tool_calls_made?.length > 0) {
        setToolBadges(res.tool_calls_made)
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        text: res.response,
        toolCalls: res.tool_calls_made || [],
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: err?.response?.data?.detail || 'An error occurred. Please try again.',
      }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, context, table, onPendingCharts])

  return (
    <div className="fixed right-0 top-0 h-full w-[26rem] bg-sentinel-surface border-l border-sentinel-border flex flex-col shadow-2xl z-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sentinel-border flex-shrink-0 bg-sentinel-surface/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-sentinel-blue/15 border border-sentinel-blue/25 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-sentinel-blue" />
          </div>
          <div>
            <div className="text-sm font-semibold text-sentinel-text">Anomaly AI</div>
            <div className="text-[10px] text-sentinel-faint">LangGraph ReAct Agent</div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 no-scrollbar">
        {messages.map((m, i) => (
          <div key={i} className={clsx('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
            {m.role === 'user' ? (
              <div className="max-w-[85%] px-3 py-2 rounded-xl bg-sentinel-blue text-white text-sm leading-relaxed">
                {m.text}
              </div>
            ) : (
              <div className="max-w-[95%] space-y-1.5">
                <div className="px-3 py-2.5 rounded-xl bg-sentinel-card border border-sentinel-border">
                  <MarkdownMessage content={m.text} />
                </div>
                {m.toolCalls?.length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1">
                    {[...new Set(m.toolCalls)].map(t => (
                      <span
                        key={t}
                        className="px-2 py-0.5 rounded-full bg-sentinel-hover border border-sentinel-border text-[10px] text-sentinel-faint font-mono"
                      >
                        {t}()
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-sentinel-card border border-sentinel-border px-4 py-3 rounded-xl flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-sentinel-blue" />
              <span className="text-xs text-sentinel-faint">Analysing...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-sentinel-border flex-shrink-0 bg-sentinel-surface/80 backdrop-blur-sm">
        <div className="flex items-end gap-2 bg-sentinel-card border border-sentinel-border rounded-xl px-3 py-2 focus-within:border-sentinel-blue/50 transition-colors">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Ask about anomalies, run SQL, generate charts..."
            rows={1}
            disabled={loading}
            className="flex-1 bg-transparent resize-none outline-none text-sm text-sentinel-text placeholder-sentinel-faint leading-relaxed py-0.5"
            style={{ maxHeight: 80 }}
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className={clsx(
              'flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all',
              input.trim() && !loading
                ? 'bg-sentinel-blue text-white hover:bg-blue-500 active:scale-95'
                : 'bg-sentinel-hover text-sentinel-faint cursor-not-allowed'
            )}
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
          </button>
        </div>
        <p className="text-[10px] text-sentinel-faint/60 text-center mt-1.5">
          Can run SQL · compute stats · generate charts · explain anomalies
        </p>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AISuggestionPanel — preview + approve AI-generated charts before injection
// ─────────────────────────────────────────────────────────────────────────────
function AISuggestionPanel({ suggestion, onAccept, onDismiss }) {
  const [note, setNote] = useState(suggestion.note ?? '')
  const iframeRef = useRef(null)

  useEffect(() => {
    const chart = suggestion.charts?.[0]
    if (!chart?.html || !iframeRef.current) return
    const doc = iframeRef.current.contentDocument || iframeRef.current.contentWindow?.document
    if (!doc) return
    doc.open()
    doc.write(`<!doctype html><html><head><meta charset="utf-8">
      <style>* { box-sizing:border-box; } body { margin:0; background:transparent; overflow:hidden; }</style>
    </head><body>${chart.html}</body></html>`)
    doc.close()
  }, [suggestion.charts])

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center"
      style={{ backdropFilter: 'blur(8px)', background: 'rgba(10,12,20,0.80)' }}>
      <div className="w-full max-w-2xl mx-4 bg-sentinel-surface border border-violet-500/40 rounded-2xl shadow-2xl overflow-hidden"
        style={{ boxShadow: '0 0 60px rgba(139,92,246,0.12)' }}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-sentinel-border bg-violet-500/5">
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-violet-400" />
            <span className="text-sm font-semibold text-sentinel-text">AI-Generated Chart — Review Before Adding</span>
            <span className="text-xs text-sentinel-faint">({suggestion.charts?.length} chart{suggestion.charts?.length !== 1 ? 's' : ''})</span>
          </div>
          <button onClick={onDismiss} className="p-1.5 rounded-lg hover:bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Chart preview */}
        {suggestion.charts?.[0]?.html && (
          <div className="border-b border-sentinel-border">
            <iframe ref={iframeRef} className="w-full" style={{ height: 260, border: 'none', display: 'block' }}
              title="AI chart preview" sandbox="allow-scripts allow-same-origin" />
          </div>
        )}
        {suggestion.charts?.length > 1 && (
          <div className="px-5 py-2 text-xs text-sentinel-faint border-b border-sentinel-border bg-sentinel-card/40">
            + {suggestion.charts.length - 1} more chart{suggestion.charts.length - 1 !== 1 ? 's' : ''} will also be added
          </div>
        )}

        {/* Editable note */}
        <div className="px-5 py-3 border-b border-sentinel-border">
          <label className="text-[10px] font-semibold text-sentinel-faint uppercase tracking-wider block mb-1.5">
            Add a note (optional)
          </label>
          <textarea
            value={note}
            onChange={e => setNote(e.target.value)}
            rows={2}
            placeholder="Describe why this chart was added, or edit the AI's suggestion..."
            className="w-full bg-sentinel-card border border-sentinel-border rounded-lg px-3 py-2 text-xs text-sentinel-text placeholder-sentinel-faint outline-none focus:border-violet-500/50 resize-none"
          />
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 px-5 py-3">
          <button onClick={onDismiss}
            className="px-4 py-2 rounded-lg text-sm text-sentinel-muted hover:text-sentinel-text hover:bg-sentinel-hover border border-sentinel-border transition-colors">
            Dismiss
          </button>
          <button onClick={() => onAccept(note)}
            className="px-4 py-2 rounded-lg text-sm bg-violet-500/20 border border-violet-500/40 text-violet-300 hover:bg-violet-500/30 font-medium transition-colors">
            Add to Dashboard
          </button>
        </div>
      </div>
    </div>
  )
}

export default function AnomalyDashboard() {
  const {
    datasets, hasModified,
    anomalyResult, setAnomalyResult,
    anomalyAgentCharts, setAnomalyAgentCharts,
    anomalyChatMessages, setAnomalyChatMessages,
  } = useApp()

  const originalTable = useMemo(() => {
    const real = datasets.filter(d => d.filename !== 'modified.csv')
    return real[0]?.tables?.[0] ?? ''
  }, [datasets])

  const modifiedTable = useMemo(() => {
    const mod = datasets.find(d => d.filename === 'modified.csv')
    return mod?.tables?.[0] ?? ''
  }, [datasets])

  const [version, setVersion]         = useState('original')
  const [threshold, setThreshold]     = useState(2.0)
  const [running, setRunning]         = useState(false)
  const [error, setError]             = useState(null)
  const [chatOpen, setChatOpen]       = useState(false)
  const [insightsOpen, setInsightsOpen] = useState(true)
  const [activeMethod, setActiveMethod] = useState('ensemble')
  const [pendingSuggestion, setPendingSuggestion] = useState(null)
  const autoRanRef   = useRef(false)
  const prevVersionRef = useRef(version)

  // Use context-persisted state so results survive navigation
  const result    = anomalyResult
  const setResult = setAnomalyResult
  const agentCharts    = anomalyAgentCharts
  const setAgentCharts = setAnomalyAgentCharts

  const activeTable = version === 'modified' ? modifiedTable : originalTable

  const handleRun = useCallback(async (tbl) => {
    const target = tbl ?? activeTable
    if (!target || running) return
    setRunning(true)
    setError(null)
    setResult(null)
    setAgentCharts([])
    setActiveMethod('ensemble')
    try {
      const res = await detectAnomalies({ table: target, threshold, method: 'ensemble' })
      setResult(res)
    } catch (err) {
      setError(err?.response?.data?.detail || 'Detection failed. Please try again.')
    } finally {
      setRunning(false)
    }
  }, [activeTable, running, threshold, setResult, setAgentCharts])

  // Auto-run once — only if no result yet (survives navigation re-mounts)
  useEffect(() => {
    if (originalTable && !autoRanRef.current && !result) {
      autoRanRef.current = true
      handleRun(originalTable)
    }
  }, [originalTable]) // eslint-disable-line

  useEffect(() => {
    if (prevVersionRef.current === version) return
    prevVersionRef.current = version
    const tbl = version === 'modified' ? modifiedTable : originalTable
    if (tbl) handleRun(tbl)
  }, [version]) // eslint-disable-line

  // Active-method anomaly records (client-side toggle — no API call)
  const activeAnomalies = useMemo(() => {
    if (!result) return []
    const mData = result.methods?.[activeMethod]
    return mData?.anomalies ?? result.anomalies ?? []
  }, [result, activeMethod])

  const activeStats = useMemo(() => {
    if (!result) return result?.stats
    return result.methods?.[activeMethod]?.stats ?? result.stats
  }, [result, activeMethod])

  const allCharts = useMemo(() => [
    ...(result?.charts ?? []),
    ...agentCharts,
  ], [result, agentCharts])

  // Called by ChatPanel when agent returns charts — show approval panel first
  const handlePendingCharts = useCallback((suggestion) => {
    setPendingSuggestion(suggestion)
  }, [])

  const handleAcceptSuggestion = useCallback((note) => {
    if (!pendingSuggestion) return
    setAgentCharts(prev => [
      ...prev,
      ...pendingSuggestion.charts.map(c => ({ ...c, agent_generated: true, note })),
    ])
    setPendingSuggestion(null)
  }, [pendingSuggestion, setAgentCharts])

  return (
    <div className="flex flex-col h-full bg-sentinel-bg overflow-hidden">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-sentinel-border bg-sentinel-surface">
        <div className="flex items-center justify-between flex-wrap gap-4">
          {/* Title */}
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-sentinel-text">Anomaly Detection</h1>
              <p className="text-xs text-sentinel-faint">
                Statistical · ML · Time-Series · Ensemble — parallel pipeline
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">Dataset</label>
              <VersionToggle version={version} setVersion={setVersion} hasModified={hasModified} />
            </div>

            {activeTable && (
              <div className="flex flex-col gap-0.5 mt-4">
                <span className="text-[10px] text-sentinel-faint">
                  Table: <span className="text-sentinel-cyan font-mono">{activeTable}</span>
                </span>
              </div>
            )}

            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">Threshold: {threshold.toFixed(1)}σ</label>
              <input
                type="range" min="1.0" max="4.0" step="0.1"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                className="w-28 accent-sentinel-blue"
              />
            </div>

            <button
              onClick={() => handleRun()}
              disabled={!activeTable || running}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all mt-4',
                activeTable && !running
                  ? 'bg-sentinel-blue text-white hover:bg-blue-500 active:scale-95'
                  : 'bg-sentinel-hover text-sentinel-faint cursor-not-allowed'
              )}
            >
              {running
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : result ? <RefreshCw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {running ? 'Running...' : result ? 'Re-run' : 'Run Detection'}
            </button>

            {result && (
              <button
                onClick={() => setChatOpen(v => !v)}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all mt-4',
                  chatOpen
                    ? 'bg-sentinel-blue/20 border border-sentinel-blue/30 text-sentinel-blue'
                    : 'bg-sentinel-card border border-sentinel-border text-sentinel-muted hover:text-sentinel-text'
                )}
              >
                <MessageSquare className="w-4 h-4" />
                Ask AI
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div
        className="flex-1 overflow-y-auto p-6 no-scrollbar space-y-6"
        style={{ paddingRight: chatOpen ? '27rem' : undefined }}
      >

        {/* Empty state */}
        {!result && !running && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <div className="p-8 rounded-2xl bg-sentinel-card border border-sentinel-border">
              <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
              <div className="text-sentinel-text font-semibold mb-2">Run Anomaly Detection</div>
              <div className="text-sm text-sentinel-faint max-w-xs">
                Upload a dataset and click{' '}
                <span className="text-sentinel-blue font-medium">Run Detection</span>{' '}
                to scan using the 4-layer hybrid pipeline (Statistical · ML · Time-Series · Ensemble).
              </div>
            </div>
          </div>
        )}

        {/* Spinner */}
        {running && (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Loader2 className="w-10 h-10 text-sentinel-blue animate-spin" />
            <div className="text-sm text-sentinel-muted">Running 3-layer parallel pipeline...</div>
            <div className="flex gap-3 text-xs text-sentinel-faint">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />Statistical
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '0.2s' }} />ML
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" style={{ animationDelay: '0.4s' }} />Time-Series
              </span>
            </div>
          </div>
        )}

        {/* Results */}
        {result && !running && (
          <>
            {/* ── Stat cards ─── */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total Anomalies" value={activeStats?.total}   icon={Activity}    colorClass="bg-sentinel-blue/10 text-sentinel-blue" />
              <StatCard label="Critical"         value={activeStats?.critical} icon={ShieldAlert}  colorClass="bg-red-500/10 text-red-400" />
              <StatCard label="High"             value={activeStats?.high}    icon={TrendingDown} colorClass="bg-amber-500/10 text-amber-400" />
              <StatCard label="Medium"           value={activeStats?.medium}  icon={ShieldCheck}  colorClass="bg-blue-500/10 text-blue-400" />
            </div>

            {/* ── Method toggle ─── */}
            {result.methods && (
              <div className="bg-sentinel-card border border-sentinel-border rounded-xl px-4 py-3 flex items-center justify-between flex-wrap gap-3">
                <div>
                  <div className="text-xs font-semibold text-sentinel-text mb-0.5">Detection Method View</div>
                  <div className="text-[11px] text-sentinel-faint">Switch to view anomalies by individual method — no re-run required</div>
                </div>
                <MethodToggle
                  activeMethod={activeMethod}
                  setActiveMethod={setActiveMethod}
                  methods={result.methods}
                />
              </div>
            )}

            {/* ── AI Insights ─── */}
            {result.insights && (
              <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden">
                <button
                  onClick={() => setInsightsOpen(v => !v)}
                  className="w-full flex items-center justify-between px-4 py-3 border-b border-sentinel-border hover:bg-sentinel-hover/30 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-sentinel-blue" />
                    <span className="text-sm font-semibold text-sentinel-text">AI Insights</span>
                  </div>
                  {insightsOpen
                    ? <ChevronUp className="w-4 h-4 text-sentinel-faint" />
                    : <ChevronDown className="w-4 h-4 text-sentinel-faint" />}
                </button>
                {insightsOpen && (
                  <div className="p-4">
                    <MarkdownMessage content={result.insights} />
                  </div>
                )}
              </div>
            )}

            {/* ── Charts grid ─── */}
            {allCharts.length > 0 && (
              <div>
                <div className="text-xs text-sentinel-faint mb-3 px-1">
                  {result.charts?.length ?? 0} detection chart{(result.charts?.length ?? 0) !== 1 ? 's' : ''}
                  {agentCharts.length > 0 && ` · ${agentCharts.length} AI-generated`}
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {allCharts.map((chart, i) => (
                    <ChartCard
                      key={`${chart.title}-${i}`}
                      chart={chart}
                      agentGenerated={chart.agent_generated === true}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* ── Anomaly table ─── */}
            {activeAnomalies.length > 0 && (
              <AnomalyTable anomalies={activeAnomalies} table={result.table} />
            )}

            {activeStats?.total === 0 && (
              <div className="text-center py-16 text-sentinel-faint">
                <ShieldCheck className="w-12 h-12 mx-auto mb-3 text-emerald-500" />
                <div className="text-sm font-semibold text-sentinel-text mb-1">No anomalies detected</div>
                <div className="text-xs">Try lowering the threshold to surface more subtle patterns.</div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Chat panel ─────────────────────────────────────────────────── */}
      {chatOpen && result && (
        <ChatPanel
          onClose={() => setChatOpen(false)}
          context={{
            table:    result.table,
            stats:    result.methods?.[activeMethod]?.stats ?? result.stats,
            profile:  result.profile,
            anomalies: result.anomalies?.slice(0, 10) ?? [],
          }}
          table={result.table}
          onPendingCharts={handlePendingCharts}
          messages={anomalyChatMessages}
          setMessages={setAnomalyChatMessages}
        />
      )}

      {/* AI suggestion approval overlay */}
      {pendingSuggestion && (
        <AISuggestionPanel
          suggestion={pendingSuggestion}
          onAccept={handleAcceptSuggestion}
          onDismiss={() => setPendingSuggestion(null)}
        />
      )}
    </div>
  )
}
