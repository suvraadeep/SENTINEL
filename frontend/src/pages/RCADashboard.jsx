import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import clsx from 'clsx'
import {
  GitBranch, Play, Loader2, MessageSquare, X, Send,
  ChevronDown, ChevronUp, ChevronRight, AlertCircle,
  TrendingDown, Activity, Network, BarChart2, Clock,
  Info, CheckCircle, RefreshCw, Layers, HelpCircle,
  Maximize2, Bot, Sparkles, Cpu, Check, Pencil,
} from 'lucide-react'
import { useApp } from '../context/AppContext'
import { runRCA, rcaChat, rcaTraverse } from '../api/client'
import MarkdownMessage from '../components/MarkdownMessage'

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function Badge({ children, color = 'blue' }) {
  const cls = {
    blue:   'bg-blue-500/15 text-blue-400 border-blue-500/30',
    amber:  'bg-amber-500/15 text-amber-400 border-amber-500/30',
    red:    'bg-red-500/15 text-red-400 border-red-500/30',
    green:  'bg-green-500/15 text-green-400 border-green-500/30',
    purple: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
    cyan:   'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    faint:  'bg-sentinel-hover text-sentinel-faint border-sentinel-border',
  }[color] ?? 'bg-sentinel-hover text-sentinel-faint border-sentinel-border'
  return (
    <span className={clsx('px-2 py-0.5 rounded-full text-[10px] font-semibold border', cls)}>
      {children}
    </span>
  )
}

function StatCard({ label, value, sub, icon: Icon, color }) {
  const ring = {
    blue:   'bg-blue-500/10 text-blue-400',
    amber:  'bg-amber-500/10 text-amber-400',
    purple: 'bg-purple-500/10 text-purple-400',
    green:  'bg-green-500/10 text-green-400',
  }[color] ?? 'bg-blue-500/10 text-blue-400'
  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-4 flex items-center gap-3 hover:border-sentinel-purple/20 transition-colors">
      <div className={clsx('p-2 rounded-lg flex-shrink-0', ring)}><Icon className="w-4 h-4" /></div>
      <div className="min-w-0">
        <div className="text-xl font-bold text-sentinel-text truncate">{value ?? '—'}</div>
        <div className="text-xs text-sentinel-faint">{label}</div>
        {sub && <div className="text-[10px] text-sentinel-faint/70 truncate">{sub}</div>}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MethodToggle
// ─────────────────────────────────────────────────────────────────────────────
function MethodToggle({ activeMethod, setActiveMethod, methods }) {
  const DEFS = [
    { key: 'statistical', label: 'Statistical', color: 'border-violet-500/50 bg-violet-500/10 text-violet-300' },
    { key: 'temporal',    label: 'Temporal',    color: 'border-amber-500/50 bg-amber-500/10 text-amber-300' },
    { key: 'graph',       label: 'Graph',       color: 'border-cyan-500/50 bg-cyan-500/10 text-cyan-300' },
    { key: 'ensemble',    label: 'Ensemble',    color: 'border-blue-500/50 bg-blue-500/10 text-blue-300' },
  ]
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {DEFS.map(({ key, label, color }) => {
        const causes = methods?.[key]?.root_causes
        const count  = Array.isArray(causes) ? causes.length : null
        const active = activeMethod === key
        return (
          <button key={key} onClick={() => setActiveMethod(key)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all duration-150',
              active ? color : 'border-sentinel-border bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted hover:border-sentinel-muted/30'
            )}>
            {label}
            {count != null && (
              <span className={clsx('px-1.5 py-0.5 rounded-full text-[10px] font-bold', active ? 'bg-white/20' : 'bg-sentinel-hover')}>
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
// ChartCard — chart + collapsible "Why?" explanation + expand
// ─────────────────────────────────────────────────────────────────────────────
function ChartCard({ chart, height = 320, agentGenerated = false }) {
  const [explOpen, setExplOpen] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const iframeRef = useRef(null)

  useEffect(() => {
    const iframe = iframeRef.current
    if (!iframe || !chart.html) return
    const doc = iframe.contentDocument || iframe.contentWindow?.document
    if (!doc) return
    doc.open()
    doc.write(`<!doctype html><html><head><meta charset="utf-8">
      <style>*{box-sizing:border-box;}body{margin:0;background:transparent;overflow:hidden;}.plotly-graph-div{width:100%!important;}</style>
    </head><body>${chart.html}</body></html>`)
    doc.close()
  }, [chart.html, expanded])

  const expl    = chart.explanation
  const hasExpl = expl && (expl.why || expl.what || expl.how_to_read)

  return (
    <div className={clsx('bg-sentinel-card border rounded-xl overflow-hidden transition-all group',
      agentGenerated ? 'border-violet-500/40' : 'border-sentinel-border hover:border-sentinel-purple/25')}>
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-sentinel-border bg-sentinel-surface/50">
        <div className="flex items-center gap-2 min-w-0">
          <BarChart2 className="w-3.5 h-3.5 text-sentinel-faint flex-shrink-0" />
          <span className="text-xs font-semibold text-sentinel-text truncate">{chart.title}</span>
          {agentGenerated && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-[10px] font-medium flex-shrink-0">
              <Bot className="w-2.5 h-2.5" />Added by AI
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          {hasExpl && (
            <button onClick={() => setExplOpen(v => !v)} title="Why this chart?"
              className={clsx('p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1',
                explOpen ? 'bg-sentinel-purple/20 text-sentinel-purple' : 'text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover')}>
              <HelpCircle className="w-3.5 h-3.5" /><span className="hidden sm:inline">Why?</span>
            </button>
          )}
          <button onClick={() => setExpanded(v => !v)} className="p-1.5 rounded-lg text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover transition-colors">
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
      {explOpen && hasExpl && (
        <div className="px-4 py-3 bg-sentinel-purple/5 border-b border-sentinel-purple/20 space-y-2">
          {expl.why && <div><span className="text-[10px] font-bold text-sentinel-purple uppercase tracking-wider">Why this chart</span><p className="text-xs text-sentinel-muted mt-0.5">{expl.why}</p></div>}
          {expl.what && <div><span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">What it shows</span><p className="text-xs text-sentinel-muted mt-0.5">{expl.what}</p></div>}
          {expl.how_to_read && <div><span className="text-[10px] font-bold text-amber-500 uppercase tracking-wider">How to read it</span><p className="text-xs text-sentinel-muted mt-0.5">{expl.how_to_read}</p></div>}
        </div>
      )}
      <iframe ref={iframeRef} className="w-full block" style={{ height: expanded ? 500 : height, border: 'none' }}
        title={chart.title} sandbox="allow-scripts allow-same-origin" />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// AISuggestionPanel — approval step before injecting agent charts
// ─────────────────────────────────────────────────────────────────────────────
function AISuggestionPanel({ pending, onAccept, onDismiss }) {
  const [note, setNote] = useState(pending.note || '')

  return (
    <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Sparkles className="w-4 h-4 text-violet-400" />
        <span className="text-sm font-semibold text-violet-300">AI Suggestion — {pending.charts.length} chart{pending.charts.length !== 1 ? 's' : ''} to add</span>
      </div>

      {/* Editable note */}
      <div>
        <label className="text-[10px] text-sentinel-faint uppercase tracking-wider block mb-1">Add a note (optional)</label>
        <textarea
          value={note}
          onChange={e => setNote(e.target.value)}
          rows={2}
          placeholder="Describe why you're adding these charts..."
          className="w-full bg-sentinel-card border border-sentinel-border rounded-lg px-3 py-2 text-xs text-sentinel-text placeholder-sentinel-faint resize-none outline-none focus:border-violet-500/50"
        />
      </div>

      {/* Chart previews */}
      {pending.charts.length > 0 && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {pending.charts.slice(0, 4).map((c, i) => (
            <ChartCard key={i} chart={c} height={200} agentGenerated />
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={() => onAccept(note)}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-violet-500 text-white text-xs font-semibold hover:bg-violet-400 active:scale-95 transition-all"
        >
          <Check className="w-3.5 h-3.5" />Accept &amp; Add to Dashboard
        </button>
        <button
          onClick={onDismiss}
          className="px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border text-xs text-sentinel-faint hover:text-sentinel-text transition-colors"
        >
          Dismiss
        </button>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// RootCauseCard
// ─────────────────────────────────────────────────────────────────────────────
function RootCauseCard({ rc, rank, onTraverse }) {
  const [expanded, setExpanded] = useState(false)
  const barW     = Math.round((rc.influence_score ?? 0) * 100)
  const barColor = rc.is_causal ? '#F59E0B' : rc.is_anomalous ? '#EF4444' : '#3B82F6'

  return (
    <div className={clsx('bg-sentinel-card border rounded-xl overflow-hidden transition-all',
      rank === 0 ? 'border-amber-500/30' : 'border-sentinel-border')}>
      <div className="flex items-center gap-3 px-4 py-3">
        <div className={clsx('w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0',
          rank === 0 ? 'bg-amber-500/15 text-amber-400' : rank === 1 ? 'bg-blue-500/15 text-blue-400' : 'bg-sentinel-hover text-sentinel-faint')}>
          {rank + 1}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-sentinel-text">{rc.name || rc.column}</span>
            {rc.is_causal    && <Badge color="amber">Granger causal</Badge>}
            {rc.is_anomalous && <Badge color="red">Anomalous</Badge>}
            {!rc.is_causal && !rc.is_anomalous && <Badge color="faint">correlated</Badge>}
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            <div className="flex-1 h-1.5 bg-sentinel-hover rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${barW}%`, background: barColor }} />
            </div>
            <span className="text-xs text-sentinel-faint w-10 text-right">{barW}%</span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button onClick={() => onTraverse(rc.name || rc.column)}
            className="p-1.5 rounded-lg text-sentinel-faint hover:text-sentinel-purple hover:bg-sentinel-purple/10 transition-colors">
            <Network className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setExpanded(v => !v)}
            className="p-1.5 rounded-lg text-sentinel-faint hover:text-sentinel-text transition-colors">
            {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
      {expanded && (
        <div className="border-t border-sentinel-border px-4 py-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
          <Stat label="Spearman ρ"    value={rc.spearman_rho?.toFixed(4)}    good={Math.abs(rc.spearman_rho ?? 0) > 0.5} />
          <Stat label="p-value"       value={rc.p_value?.toFixed(4)}          good={(rc.p_value ?? 1) < 0.05} />
          {rc.granger_p    != null && <Stat label="Granger p"    value={rc.granger_p?.toFixed(4)}    good={rc.granger_p < 0.05} />}
          {rc.partial_corr != null && <Stat label="Partial corr" value={rc.partial_corr?.toFixed(4)} good={Math.abs(rc.partial_corr) > 0.3} />}
          {rc.distance_corr!= null && <Stat label="Distance corr" value={rc.distance_corr?.toFixed(4)} />}
          <Stat label="MI score" value={rc.mi_score?.toFixed(4)} />
          <Stat label="Depth"    value={`L${rc.depth ?? 1}`} />
          <div className="col-span-2"><span className="text-sentinel-faint">Edge type: </span><span className="text-sentinel-muted capitalize">{rc.edge_type}</span></div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, good }) {
  return (
    <div>
      <span className="text-sentinel-faint">{label}: </span>
      <span className={good === true ? 'text-green-400' : good === false ? 'text-amber-400' : 'text-sentinel-muted'}>
        {value ?? '—'}
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// StatsTable
// ─────────────────────────────────────────────────────────────────────────────
function StatsTable({ title, rows, cols }) {
  const [open, setOpen] = useState(true)
  if (!rows || rows.length === 0) return null
  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl overflow-hidden">
      <button onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 border-b border-sentinel-border hover:bg-sentinel-hover/30 transition-colors">
        <span className="text-sm font-semibold text-sentinel-text">{title}</span>
        {open ? <ChevronUp className="w-4 h-4 text-sentinel-faint" /> : <ChevronDown className="w-4 h-4 text-sentinel-faint" />}
      </button>
      {open && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-sentinel-border bg-sentinel-surface/40">
                {cols.map(c => <th key={c.key} className="px-3 py-2.5 text-left text-sentinel-faint font-medium">{c.label}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-sentinel-border/50 hover:bg-sentinel-hover/20 transition-colors">
                  {cols.map(c => {
                    const val = row[c.key]
                    return (
                      <td key={c.key} className={clsx('px-3 py-2', c.mono && 'font-mono')}>
                        {c.render ? c.render(val, row) : (typeof val === 'number' ? val.toFixed(4) : String(val ?? '—'))}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ChangePoints
// ─────────────────────────────────────────────────────────────────────────────
function ChangePoints({ cps }) {
  if (!cps || cps.length === 0) return null
  return (
    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-sentinel-purple" />
        <span className="text-sm font-semibold text-sentinel-text">Change Points Detected</span>
        <Badge color="purple">{cps.length}</Badge>
      </div>
      <div className="space-y-2">
        {cps.map((cp, i) => (
          <div key={i} className="flex items-center gap-3 text-xs">
            <div className={clsx('w-2 h-2 rounded-full flex-shrink-0', cp.direction === 'down' ? 'bg-red-400' : 'bg-green-400')} />
            <span className="text-sentinel-muted font-mono flex-shrink-0 w-32 truncate">{cp.at}</span>
            <span className={cp.direction === 'down' ? 'text-red-400' : 'text-green-400'}>
              {cp.direction === 'down' ? '▼' : '▲'} {cp.shift_sigma?.toFixed(1)}σ shift
            </span>
            <span className="text-sentinel-faint">{cp.left_mean?.toFixed(2)} → {cp.right_mean?.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TraverseModal
// ─────────────────────────────────────────────────────────────────────────────
function TraverseModal({ feature, table, targetCol, onClose }) {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr]         = useState(null)

  useEffect(() => {
    rcaTraverse({ table, feature, target_col: targetCol })
      .then(r => { setResult(r); setLoading(false) })
      .catch(e => { setErr(e?.response?.data?.detail || 'Traversal failed'); setLoading(false) })
  }, [feature, table, targetCol])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-sentinel-surface border border-sentinel-border rounded-2xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-sentinel-border">
          <div className="flex items-center gap-2">
            <Network className="w-4 h-4 text-sentinel-purple" />
            <span className="text-sm font-bold text-sentinel-text">Upstream causes of <em>{feature}</em></span>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-5 max-h-[70vh] overflow-y-auto no-scrollbar">
          {loading && <div className="flex justify-center py-8"><Loader2 className="w-6 h-6 animate-spin text-sentinel-faint" /></div>}
          {err    && <div className="text-sm text-red-400">{err}</div>}
          {result && !loading && (
            <div className="space-y-4">
              {result.explanation && (
                <div className="p-3 rounded-xl bg-sentinel-purple/5 border border-sentinel-purple/20">
                  <MarkdownMessage content={result.explanation} />
                </div>
              )}
              {result.causal_chain?.length > 0 ? (
                <div className="space-y-2">
                  <div className="text-xs text-sentinel-faint font-semibold uppercase tracking-wider">Causal chain</div>
                  {result.causal_chain.map((rc, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-sentinel-card border border-sentinel-border text-xs">
                      <div className="w-5 h-5 rounded-full bg-blue-500/15 text-blue-400 flex items-center justify-center font-bold flex-shrink-0">{i+1}</div>
                      <div className="flex-1">
                        <div className="font-medium text-sentinel-text">{rc.name}</div>
                        <div className="text-sentinel-faint">ρ={rc.spearman_rho?.toFixed(3)}, influence={rc.influence_score?.toFixed(3)}</div>
                      </div>
                      {rc.is_causal && <Badge color="amber">causal</Badge>}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-sentinel-faint text-center py-4">No upstream causes found.</div>
              )}
            </div>
          )}
        </div>
      </div>
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
      <div className={clsx('flex rounded-lg overflow-hidden border-2 transition-all duration-300',
        version === 'modified' && hasModified ? 'border-cyan-500/60' : 'border-indigo-500/40')}>
        {['original', 'modified'].map(v => (
          <button key={v} onClick={() => hasModified && setVersion(v)} disabled={v === 'modified' && !hasModified}
            className={clsx('px-3.5 py-1.5 text-xs font-bold transition-all duration-200 capitalize',
              version === v
                ? v === 'modified' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-indigo-500/20 text-indigo-300'
                : v === 'modified' && !hasModified ? 'bg-sentinel-card text-sentinel-faint/40 cursor-not-allowed'
                : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted cursor-pointer')}>
            {v === 'original' ? '⬡ Original' : '⬢ Modified'}
          </button>
        ))}
      </div>
      {!hasModified && <span className="text-[9px] text-sentinel-faint/50 italic">No modified version</span>}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ChatDrawer
// ─────────────────────────────────────────────────────────────────────────────
function ChatDrawer({ onClose, context, onPendingCharts, messages, setMessages }) {
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [traversalPath, setTraversalPath] = useState([])
  const bottomRef = useRef(null)
  const taRef     = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
  useEffect(() => {
    if (taRef.current) {
      taRef.current.style.height = 'auto'
      taRef.current.style.height = Math.min(taRef.current.scrollHeight, 120) + 'px'
    }
  }, [input])

  const send = useCallback(async () => {
    const msg = input.trim()
    if (!msg || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setLoading(true)
    const history = messages.filter(m => m.role !== 'system').slice(-10).map(m => ({ role: m.role, content: m.text }))
    try {
      const res = await rcaChat({ message: msg, context, table: context?.table, chat_history: history })
      if (res.charts?.length > 0) {
        // Send charts to parent for approval — don't inject directly
        onPendingCharts?.({ charts: res.charts, note: res.response?.slice(0, 120) })
      }
      if (res.traversal_path?.length) setTraversalPath(res.traversal_path)
      setMessages(prev => [...prev, { role: 'assistant', text: res.response, toolCalls: res.tool_calls_made || [] }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: err?.response?.data?.detail || 'An error occurred.' }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, context, onPendingCharts])

  const QUICK_Q = [
    `Why did ${context?.target_col || 'the metric'} change?`,
    'Explain the top root cause',
    'Show the causal chain',
    'What actions should I take?',
  ]

  return (
    <div className="fixed right-0 top-0 h-full w-[26rem] bg-sentinel-surface border-l border-sentinel-border flex flex-col shadow-2xl z-50">
      <div className="flex items-center justify-between px-4 py-3 border-b border-sentinel-border flex-shrink-0 bg-sentinel-surface/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-sentinel-purple/15 border border-sentinel-purple/25 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-sentinel-purple" />
          </div>
          <div>
            <div className="text-sm font-semibold text-sentinel-text">RCA Assistant</div>
            <div className="text-[10px] text-sentinel-faint">LangGraph ReAct · graph-aware</div>
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text transition-colors"><X className="w-4 h-4" /></button>
      </div>

      {traversalPath.length > 0 && (
        <div className="flex items-center gap-1 px-4 py-2 border-b border-sentinel-border bg-sentinel-purple/5 flex-shrink-0 flex-wrap">
          <span className="text-[10px] text-sentinel-purple font-semibold uppercase tracking-wider mr-1">Path:</span>
          {traversalPath.map((p, i) => (
            <React.Fragment key={i}>
              <span className="text-[10px] text-sentinel-muted font-mono">{p}</span>
              {i < traversalPath.length - 1 && <ChevronRight className="w-2.5 h-2.5 text-sentinel-faint" />}
            </React.Fragment>
          ))}
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-3 no-scrollbar">
        {messages.map((m, i) => (
          <div key={i} className={clsx('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
            {m.role === 'user' ? (
              <div className="max-w-[85%] px-3 py-2 rounded-xl bg-sentinel-purple text-white text-sm leading-relaxed">{m.text}</div>
            ) : (
              <div className="max-w-[95%] space-y-1.5">
                <div className="px-3 py-2.5 rounded-xl bg-sentinel-card border border-sentinel-border">
                  <MarkdownMessage content={m.text} />
                </div>
                {m.toolCalls?.length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1">
                    {[...new Set(m.toolCalls)].map(t => (
                      <span key={t} className="px-2 py-0.5 rounded-full bg-sentinel-hover border border-sentinel-border text-[10px] text-sentinel-faint font-mono">{t}()</span>
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
              <Loader2 className="w-3.5 h-3.5 animate-spin text-sentinel-purple" />
              <span className="text-xs text-sentinel-faint">Traversing causal graph…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-4 pt-2 flex flex-wrap gap-1.5 flex-shrink-0">
        {QUICK_Q.map(q => (
          <button key={q} onClick={() => { setInput(q); taRef.current?.focus() }}
            className="text-[10px] px-2 py-1 rounded-full bg-sentinel-hover text-sentinel-faint hover:text-sentinel-text hover:bg-sentinel-border transition-colors">
            {q}
          </button>
        ))}
      </div>

      <div className="p-3 border-t border-sentinel-border flex-shrink-0 bg-sentinel-surface/80 backdrop-blur-sm">
        <div className="flex items-end gap-2 bg-sentinel-card border border-sentinel-border rounded-xl px-3 py-2 focus-within:border-sentinel-purple/40 transition-colors">
          <textarea ref={taRef} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Ask about root causes…" rows={1} disabled={loading}
            className="flex-1 bg-transparent resize-none outline-none text-sm text-sentinel-text placeholder-sentinel-faint leading-relaxed py-0.5"
            style={{ maxHeight: 120 }} />
          <button onClick={send} disabled={!input.trim() || loading}
            className={clsx('flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all',
              input.trim() && !loading ? 'bg-sentinel-purple text-white hover:bg-purple-500 active:scale-95' : 'bg-sentinel-hover text-sentinel-faint cursor-not-allowed')}>
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
          </button>
        </div>
        <p className="text-[10px] text-sentinel-faint/60 text-center mt-1.5">Can traverse graph · explain causes · generate charts · run SQL</p>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// RCADashboard — main page
// ─────────────────────────────────────────────────────────────────────────────
export default function RCADashboard() {
  const {
    datasets, hasModified,
    rcaResult, setRcaResult,
    rcaAgentCharts, setRcaAgentCharts,
    rcaChatMessages, setRcaChatMessages,
  } = useApp()

  const originalTable = useMemo(() => datasets.filter(d => d.filename !== 'modified.csv')[0]?.tables?.[0] ?? '', [datasets])
  const modifiedTable = useMemo(() => datasets.find(d => d.filename === 'modified.csv')?.tables?.[0] ?? '', [datasets])

  const [version, setVersion]           = useState('original')
  const [targetCol, setTargetCol]       = useState('')
  const [pThreshold, setPThreshold]     = useState(0.05)
  const [topK, setTopK]                 = useState(8)
  const [running, setRunning]           = useState(false)
  const [error, setError]               = useState(null)
  const [chatOpen, setChatOpen]         = useState(false)
  const [traverseFeature, setTraverse]  = useState(null)
  const [activeTab, setActiveTab]       = useState('graph')
  const [insightsOpen, setInsightsOpen] = useState(true)
  // Method toggle — drives which root causes are shown (NO re-run needed)
  const [activeMethod, setActiveMethod] = useState('ensemble')
  // Pending AI suggestion (charts awaiting approval)
  const [pendingSuggestion, setPendingSuggestion] = useState(null)

  const result     = rcaResult
  const setResult  = setRcaResult
  const agentCharts    = rcaAgentCharts
  const setAgentCharts = setRcaAgentCharts

  const autoRanRef    = useRef(false)
  const prevVersionRef = useRef(version)
  const activeTable    = version === 'modified' ? modifiedTable : originalTable

  useEffect(() => {
    if (result?.profile?.target_col && !targetCol) setTargetCol(result.profile.target_col)
  }, [result]) // eslint-disable-line

  const handleRun = useCallback(async (tbl) => {
    const target = tbl ?? activeTable
    if (!target || running) return
    setRunning(true)
    setError(null)
    setResult(null)
    setAgentCharts([])
    setActiveMethod('ensemble')
    setPendingSuggestion(null)
    try {
      const res = await runRCA({ table: target, target_col: targetCol || null, p_threshold: pThreshold, top_k: topK, anomaly_context: null })
      setResult(res)
      setActiveTab('graph')
    } catch (err) {
      setError(err?.response?.data?.detail || 'RCA failed — please try again.')
    } finally {
      setRunning(false)
    }
  }, [activeTable, running, targetCol, pThreshold, topK, setResult, setAgentCharts])

  // Auto-run once — only if no persisted result
  useEffect(() => {
    if (originalTable && !autoRanRef.current && !result) {
      autoRanRef.current = true
      handleRun(originalTable)
    }
  }, [originalTable]) // eslint-disable-line

  // Re-run on version change
  useEffect(() => {
    if (prevVersionRef.current === version) return
    prevVersionRef.current = version
    const tbl = version === 'modified' ? modifiedTable : originalTable
    if (tbl) handleRun(tbl)
  }, [version]) // eslint-disable-line

  // ── Active root causes — driven by method toggle (client-side, no API call) ──
  const activeRootCauses = useMemo(() => {
    if (!result) return []
    const methodCauses = result.methods?.[activeMethod]?.root_causes
    return Array.isArray(methodCauses) ? methodCauses : (result.root_causes ?? [])
  }, [result, activeMethod])

  // Stat cards derived from activeRootCauses
  const numCauses    = activeRootCauses.length
  const primaryCause = activeRootCauses[0]
  const causalCount  = activeRootCauses.filter(r => r.is_causal).length
  const maxRho       = primaryCause ? Math.abs(primaryCause.spearman_rho ?? 0).toFixed(3) : '—'
  const usedPC       = result?.statistics?.pc_algorithm_used === true

  // Method-specific stat card #3 (replaces generic "Granger Causal" when irrelevant)
  const methodCard3 = useMemo(() => {
    if (activeMethod === 'statistical') {
      const highRho = activeRootCauses.filter(r => Math.abs(r.spearman_rho ?? 0) > 0.5).length
      return { label: 'High |ρ| > 0.5', value: highRho, sub: 'strong correlation', icon: Activity, color: 'green' }
    }
    if (activeMethod === 'graph') {
      const graphData = result?.methods?.graph?.graph ?? result?.graph ?? {}
      const edgeCount = (graphData.edges ?? []).length
      return { label: 'Causal Edges', value: edgeCount, sub: usedPC ? 'PC algorithm' : 'correlation DAG', icon: Network, color: 'cyan' }
    }
    return { label: 'Granger Causal', value: causalCount, sub: 'temporal precedence', icon: Clock, color: 'blue' }
  }, [activeMethod, activeRootCauses, causalCount, result, usedPC])

  const methodPrimaryCharts = useMemo(() => {
    if (!result?.charts) return []
    const chartTypeMap = {
      statistical: ['spearman_vs_partial', 'mutual_information', 'distance_correlation'],
      temporal:    ['metrics_timeline', 'lag_correlation'],
      graph:       ['causal_graph', 'root_cause_ranking'],
      ensemble:    ['causal_graph', 'root_cause_ranking'],
    }
    const preferred = chartTypeMap[activeMethod] ?? ['causal_graph']
    const found = preferred
      .map(t => result.charts.find(c => c.chart_type === t))
      .filter(Boolean)
    // Fallback: show first chart
    return found.length > 0 ? found : result.charts.slice(0, 1)
  }, [result, activeMethod])

  const allCharts = useMemo(() => [
    ...(result?.charts ?? []),
    ...agentCharts,
  ], [result, agentCharts])

  // Chat context
  const chatContext = useMemo(() => result ? {
    table:       result.table,
    target_col:  result.target_col,
    root_causes: result.root_causes?.slice(0, 5),
    statistics:  result.statistics,
    profile:     result.profile,
    graph:       result.graph,
  } : null, [result])

  // Stat table rows
  const spearmanRows = useMemo(() => result
    ? Object.entries(result.statistics?.spearman ?? {}).map(([k, v]) => ({
        feature: k, rho: v.rho, p_value: v.p_value,
        dist_corr: result.statistics?.distance_corr?.[k] ?? result.statistics?.distance_correlation?.[k] ?? null,
      })).sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho))
    : [], [result])

  const grangerRows = useMemo(() => result
    ? Object.entries(result.statistics?.granger ?? {}).map(([k, v]) => ({
        feature: k, p_value: typeof v === 'number' ? v : v?.p_value ?? 1,
        significant: (typeof v === 'number' ? v : v?.p_value ?? 1) < pThreshold,
      })).sort((a, b) => a.p_value - b.p_value)
    : [], [result, pThreshold])

  const miRows = useMemo(() => result
    ? Object.entries(result.statistics?.mutual_information ?? {}).map(([k, v]) => ({ feature: k, mi: v }))
        .sort((a, b) => b.mi - a.mi)
    : [], [result])

  const TABS = [
    { id: 'graph',  label: 'Causal Graph', icon: Network },
    { id: 'causes', label: 'Root Causes',  icon: GitBranch },
    { id: 'charts', label: 'Charts',       icon: BarChart2 },
    { id: 'stats',  label: 'Statistics',   icon: Activity },
  ]

  return (
    <div className="flex flex-col h-full bg-sentinel-bg overflow-hidden" style={{ paddingRight: chatOpen ? '27rem' : 0 }}>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex-shrink-0 bg-sentinel-surface border-b border-sentinel-border px-6 py-4">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-sentinel-purple/10 border border-sentinel-purple/20">
              <GitBranch className="w-5 h-5 text-sentinel-purple" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-sentinel-text">Root Cause Analysis</h1>
              <p className="text-xs text-sentinel-faint">Bulk Spearman · Distance corr · PC Algorithm · Granger (top-2) · PageRank</p>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">Dataset</label>
              <VersionToggle version={version} setVersion={setVersion} hasModified={hasModified} />
            </div>
            {activeTable && (
              <div className="flex flex-col gap-0.5 mt-4">
                <span className="text-[10px] text-sentinel-faint">Table: <span className="text-sentinel-cyan font-mono">{activeTable}</span></span>
              </div>
            )}
            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">Target metric (optional)</label>
              <input value={targetCol} onChange={e => setTargetCol(e.target.value)} placeholder="auto-detect"
                className="bg-sentinel-card border border-sentinel-border rounded-lg px-2.5 py-1.5 text-xs text-sentinel-text outline-none focus:border-sentinel-purple/50 w-36 placeholder-sentinel-faint" />
            </div>
            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">p-threshold: {pThreshold}</label>
              <input type="range" min="0.01" max="0.2" step="0.01" value={pThreshold} onChange={e => setPThreshold(parseFloat(e.target.value))} className="w-24 accent-sentinel-purple" />
            </div>
            <div className="flex flex-col gap-0.5">
              <label className="text-[10px] text-sentinel-faint">Top-K: {topK}</label>
              <input type="range" min="3" max="15" step="1" value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="w-24 accent-sentinel-purple" />
            </div>
            <button onClick={() => handleRun()} disabled={!activeTable || running}
              className={clsx('flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all mt-4',
                activeTable && !running ? 'bg-sentinel-purple text-white hover:bg-purple-500 active:scale-95' : 'bg-sentinel-hover text-sentinel-faint cursor-not-allowed')}>
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : result ? <RefreshCw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {running ? 'Analyzing…' : result ? 'Re-run RCA' : 'Run RCA'}
            </button>
            {result && (
              <button onClick={() => setChatOpen(v => !v)}
                className={clsx('flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all mt-4',
                  chatOpen ? 'bg-sentinel-purple/20 border border-sentinel-purple/30 text-sentinel-purple' : 'bg-sentinel-card border border-sentinel-border text-sentinel-muted hover:text-sentinel-text')}>
                <MessageSquare className="w-4 h-4" />Ask AI
              </button>
            )}
          </div>
        </div>
        {error && (
          <div className="mt-3 flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />{error}
          </div>
        )}
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto no-scrollbar">
        {!result && !running && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center p-8 max-w-sm">
              <div className="p-5 rounded-2xl bg-sentinel-card border border-sentinel-border inline-block mb-4">
                <GitBranch className="w-12 h-12 text-sentinel-purple mx-auto" />
              </div>
              <div className="text-sentinel-text font-semibold mb-2">Causal Root Cause Analysis</div>
              <div className="text-sm text-sentinel-faint leading-relaxed">
                Select a table and click <span className="text-sentinel-purple font-medium">Run RCA</span> to build a Metric Dependency Graph.
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-sentinel-faint">
                {['Bulk Spearman','Distance corr','Mutual info','Granger (top-2)','PC Algorithm','PageRank','Causal graph','LLM explanation'].map(f => (
                  <div key={f} className="flex items-center gap-1.5 p-2 rounded-lg bg-sentinel-card border border-sentinel-border">
                    <CheckCircle className="w-3 h-3 text-sentinel-purple flex-shrink-0" />{f}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {running && (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <Loader2 className="w-10 h-10 text-sentinel-purple animate-spin" />
            <div className="text-sm text-sentinel-muted">Running causal analysis pipeline…</div>
            <div className="flex gap-3 text-xs text-sentinel-faint">
              {['Statistical','Temporal','Graph'].map((l, i) => (
                <span key={l} className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: ['#a78bfa','#f59e0b','#22d3ee'][i], animationDelay: `${i*0.2}s` }} />{l}
                </span>
              ))}
            </div>
          </div>
        )}

        {result && !running && (
          <div className="p-6 space-y-5">

            {/* Stat cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard label="Root Causes Found"  value={numCauses}    icon={GitBranch}  color="purple" />
              <StatCard label="Primary Driver" value={primaryCause?.name ?? primaryCause?.column ?? '—'}
                sub={primaryCause ? `influence ${((primaryCause.influence_score ?? 0)*100).toFixed(0)}%` : ''}
                icon={TrendingDown} color="amber" />
              <StatCard label={methodCard3.label} value={methodCard3.value} sub={methodCard3.sub}
                icon={methodCard3.icon} color={methodCard3.color} />
              <StatCard label="Max |ρ|"         value={maxRho}     sub={`target: ${result.target_col}`} icon={Activity} color="green" />
            </div>

            {/* Method toggle */}
            {result.methods && (
              <div className="bg-sentinel-card border border-sentinel-border rounded-xl px-4 py-3 flex items-center justify-between flex-wrap gap-3">
                <div>
                  <div className="text-xs font-semibold text-sentinel-text mb-0.5">Analysis Layer View</div>
                  <div className="text-[11px] text-sentinel-faint">Switch layers — updates root causes &amp; stats instantly, no re-run needed</div>
                </div>
                <MethodToggle activeMethod={activeMethod} setActiveMethod={setActiveMethod} methods={result.methods} />
              </div>
            )}

            {/* AI Suggestion Panel (pending charts) */}
            {pendingSuggestion && (
              <AISuggestionPanel
                pending={pendingSuggestion}
                onAccept={(note) => {
                  setAgentCharts(prev => [...prev, ...pendingSuggestion.charts.map(c => ({ ...c, agent_generated: true, note }))])
                  setPendingSuggestion(null)
                  setActiveTab('charts')
                }}
                onDismiss={() => setPendingSuggestion(null)}
              />
            )}

            {/* AI explanation */}
            {result.explanation && (
              <div className="bg-sentinel-card border border-sentinel-purple/20 rounded-xl overflow-hidden">
                <button onClick={() => setInsightsOpen(v => !v)}
                  className="w-full flex items-center justify-between px-4 py-3 hover:bg-sentinel-hover/20 transition-colors">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-sentinel-purple" />
                    <span className="text-sm font-semibold text-sentinel-text">Causal Explanation</span>
                    <Badge color="purple">LLM + evidence</Badge>
                  </div>
                  {insightsOpen ? <ChevronUp className="w-4 h-4 text-sentinel-faint" /> : <ChevronDown className="w-4 h-4 text-sentinel-faint" />}
                </button>
                {insightsOpen && (
                  <div className="px-4 pb-4 border-t border-sentinel-border pt-3">
                    <MarkdownMessage content={result.explanation} />
                  </div>
                )}
              </div>
            )}

            <ChangePoints cps={result.change_points} />

            {/* Tabs */}
            <div className="flex gap-1 bg-sentinel-card border border-sentinel-border rounded-xl p-1 w-fit">
              {TABS.map(t => {
                const Icon = t.icon
                const badge = t.id === 'causes' ? activeRootCauses.length : t.id === 'charts' ? allCharts.length : null
                return (
                  <button key={t.id} onClick={() => setActiveTab(t.id)}
                    className={clsx('flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                      activeTab === t.id ? 'bg-sentinel-purple/15 text-sentinel-purple' : 'text-sentinel-faint hover:text-sentinel-muted')}>
                    <Icon className="w-3.5 h-3.5" />{t.label}
                    {badge != null && badge > 0 && <span className="ml-1 px-1.5 py-0.5 rounded-full bg-sentinel-hover text-[10px]">{badge}</span>}
                  </button>
                )
              })}
            </div>

            {/* ── Tab: Causal Graph / Method View ── */}
            {activeTab === 'graph' && (
              <div className="space-y-4">
                {/* Method context banner */}
                {activeMethod === 'statistical' && (
                  <div className="flex items-start gap-3 p-3 rounded-xl bg-violet-500/10 border border-violet-500/20 text-xs text-violet-300">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Statistical view</span> — Spearman rank correlation, partial correlation (direct effects after removing confounders), mutual information (non-linear), and distance correlation. Features ranked by composite statistical score.
                    </div>
                  </div>
                )}
                {activeMethod === 'temporal' && (
                  <div className="flex items-start gap-3 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Temporal view</span> — Granger causality (VAR F-test, p&lt;{pThreshold}) and lag correlation analysis. Amber bars at lag&gt;0 indicate a driver that leads the target in time — temporal precedence evidence.
                      {!result.methods?.temporal?.has_temporal_data && (
                        <span className="ml-1 opacity-70">No date column detected — showing features with strongest temporal signal from lag analysis.</span>
                      )}
                    </div>
                  </div>
                )}
                {activeMethod === 'graph' && (
                  <div className="flex items-start gap-3 p-3 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-xs text-cyan-300">
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Graph view</span> — Causal structure from {usedPC ? 'PC algorithm (constraint-based discovery, n≤8 cols)' : 'correlation DAG (greedy topological sort)'}. BFS traversal from target with PageRank centrality boost.
                    </div>
                  </div>
                )}
                {activeMethod === 'ensemble' && (
                  <div className="flex items-start gap-3 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-xs text-blue-300">
                    <Sparkles className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold">Ensemble view</span> — All layers combined: Statistical (Spearman + MI + dCor) + Temporal (Granger + lag) + Graph (BFS + PageRank). Composite influence score = |ρ| + partial_corr + MI + 0.15 if anomalous.
                    </div>
                  </div>
                )}

                {usedPC && (activeMethod === 'graph' || activeMethod === 'ensemble') && (
                  <span className="flex items-center gap-1 w-fit px-2 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[10px] font-semibold">
                    <Cpu className="w-3 h-3" />PC Algorithm used
                  </span>
                )}

                {/* Method-specific primary charts */}
                {methodPrimaryCharts.length > 0 ? (
                  <div className={clsx(
                    'grid gap-4',
                    methodPrimaryCharts.length > 1 && activeMethod !== 'graph' && activeMethod !== 'ensemble'
                      ? 'grid-cols-1 xl:grid-cols-2'
                      : 'grid-cols-1'
                  )}>
                    {methodPrimaryCharts.map((chart, i) => (
                      <ChartCard
                        key={`${chart.title}-${i}`}
                        chart={chart}
                        height={methodPrimaryCharts.length === 1 ? 440 : 320}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-8 text-center text-sentinel-faint text-sm">
                    {activeMethod === 'temporal'
                      ? 'No temporal charts available — dataset has no date column or fewer than 30 rows.'
                      : 'Chart not available.'}
                  </div>
                )}

                {(activeMethod === 'graph' || activeMethod === 'ensemble') && (
                  <div className="flex flex-wrap gap-4 text-xs text-sentinel-faint px-1">
                    <div className="flex items-center gap-1.5"><span className="w-8 h-0.5 bg-amber-400 inline-block rounded" />Granger causal</div>
                    <div className="flex items-center gap-1.5"><span className="w-8 h-0.5 bg-blue-500 inline-block rounded" />Correlation</div>
                    <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" />Target</div>
                    <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full bg-amber-400 inline-block" />Primary causes</div>
                  </div>
                )}
              </div>
            )}

            {/* ── Tab: Root Causes (uses activeRootCauses — driven by method toggle) ── */}
            {activeTab === 'causes' && (
              <div className="space-y-3">
                {activeRootCauses.length > 0 ? (
                  activeRootCauses.map((rc, i) => (
                    <RootCauseCard key={rc.name ?? rc.column ?? i} rc={rc} rank={i} onTraverse={feat => setTraverse(feat)} />
                  ))
                ) : (
                  <div className="text-center py-10 text-sentinel-faint text-sm">
                    No root causes found for the <strong>{activeMethod}</strong> method.
                    {activeMethod !== 'ensemble' && ' Try switching to Ensemble for combined results.'}
                  </div>
                )}
              </div>
            )}

            {/* ── Tab: Charts ── */}
            {activeTab === 'charts' && (
              <div className="space-y-4">
                {allCharts.length > 0 ? (
                  <>
                    <div className="text-xs text-sentinel-faint px-1">
                      {(result.charts?.length ?? 0)} analysis chart{(result.charts?.length ?? 0) !== 1 ? 's' : ''}
                      {agentCharts.length > 0 && ` · ${agentCharts.length} AI-generated`}
                    </div>
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                      {allCharts.map((c, i) => (
                        <ChartCard key={`${c.title}-${i}`} chart={c} height={320} agentGenerated={c.agent_generated === true} />
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-10 text-sentinel-faint text-sm">No charts yet. Ask the AI to generate one.</div>
                )}
              </div>
            )}

            {/* ── Tab: Statistics (method-aware section ordering) ── */}
            {activeTab === 'stats' && (
              <div className="space-y-4">
                {/* Method context label */}
                <div className="flex items-center gap-2 text-xs text-sentinel-faint">
                  <Layers className="w-3.5 h-3.5" />
                  <span>
                    {activeMethod === 'statistical' && 'Statistical layer — Spearman rank correlation · Mutual information · Distance correlation'}
                    {activeMethod === 'temporal'    && 'Temporal layer — Lag correlation · Granger causality · Change point detection'}
                    {activeMethod === 'graph'       && 'Graph layer — Causal structure · Spearman + partial correlation evidence'}
                    {activeMethod === 'ensemble'    && 'Ensemble — All statistical, temporal, and graph signals combined'}
                  </span>
                </div>

                {/* TEMPORAL: Lag analysis first, then Granger */}
                {(activeMethod === 'temporal' || activeMethod === 'ensemble') && (
                  <>
                    {result.statistics?.lag_analysis && Object.keys(result.statistics.lag_analysis).length > 0 && (
                      <StatsTable title="Lag Correlation Analysis (amber = driver leads target)"
                        rows={Object.entries(result.statistics.lag_analysis).map(([k, v]) => ({ feature: k, ...v }))}
                        cols={[
                          { key: 'feature',            label: 'Feature',      mono: true },
                          { key: 'optimal_lag',         label: 'Best lag',     mono: true },
                          { key: 'correlation',         label: 'Correlation',  mono: true, render: v => <span className={Math.abs(v) > 0.5 ? 'text-green-400' : 'text-sentinel-muted'}>{v?.toFixed(4)}</span> },
                          { key: 'temporal_precedence', label: 'Leads target?', render: v => <Badge color={v ? 'amber' : 'faint'}>{v ? 'Yes' : 'No'}</Badge> },
                        ]}
                      />
                    )}
                    {grangerRows.length > 0 && (
                      <StatsTable title="Granger Causality Tests (top-2 only, max_lag=2)" rows={grangerRows}
                        cols={[
                          { key: 'feature', label: 'Feature', mono: true },
                          { key: 'p_value', label: 'p-value', mono: true, render: v => <span className={v < 0.05 ? 'text-green-400' : 'text-amber-400'}>{v?.toFixed(4)}</span> },
                          { key: 'significant', label: 'Granger causal?', render: v => <Badge color={v ? 'green' : 'faint'}>{v ? 'Yes' : 'No'}</Badge> },
                        ]}
                      />
                    )}
                    {activeMethod === 'temporal' && grangerRows.length === 0 && (
                      <div className="p-4 rounded-xl bg-sentinel-card border border-sentinel-border text-xs text-sentinel-faint">
                        No Granger tests run — requires a date column and at least 30 rows.
                        {result.statistics?.lag_analysis && Object.keys(result.statistics.lag_analysis).length > 0
                          ? ' Lag correlation results shown above.'
                          : ' No temporal data available.'}
                      </div>
                    )}
                  </>
                )}

                {/* STATISTICAL (primary) or GRAPH/ENSEMBLE (secondary) */}
                <StatsTable
                  title={`Spearman Correlation${activeMethod === 'statistical' ? ' — primary statistical signal' : ''} with ${result.target_col}`}
                  rows={spearmanRows}
                  cols={[
                    { key: 'feature',   label: 'Feature',      mono: true },
                    { key: 'rho',       label: 'ρ (Spearman)', mono: true,
                      render: v => <span className={Math.abs(v) > 0.5 ? 'text-green-400' : Math.abs(v) > 0.3 ? 'text-amber-400' : 'text-sentinel-muted'}>{v?.toFixed(4)}</span> },
                    { key: 'p_value',   label: 'p-value',  mono: true,
                      render: v => <span className={v < 0.05 ? 'text-green-400' : v < 0.1 ? 'text-amber-400' : 'text-red-400'}>{v?.toFixed(4)}</span> },
                    { key: 'dist_corr', label: 'Distance corr', mono: true,
                      render: v => v != null
                        ? <span className={v > 0.4 ? 'text-cyan-400' : 'text-sentinel-muted'}>{v?.toFixed(4)}</span>
                        : <span className="text-sentinel-faint/40">—</span> },
                  ]}
                />

                {/* MI — shown for statistical and ensemble, suppressed for temporal/graph */}
                {(activeMethod === 'statistical' || activeMethod === 'ensemble') && miRows.length > 0 && (
                  <StatsTable title="Mutual Information — non-linear feature relevance I(X;Y) = H(X) − H(X|Y)" rows={miRows}
                    cols={[
                      { key: 'feature', label: 'Feature', mono: true },
                      { key: 'mi',      label: 'MI score', mono: true,
                        render: v => {
                          const maxMI = Math.max(...miRows.map(r => r.mi), 1e-9)
                          return (
                            <div className="flex items-center gap-2">
                              <div className="w-20 h-1.5 bg-sentinel-hover rounded-full overflow-hidden">
                                <div className="h-full bg-sentinel-cyan rounded-full" style={{ width: `${(v / maxMI) * 100}%` }} />
                              </div>
                              <span className="text-sentinel-muted font-mono">{v?.toFixed(4)}</span>
                            </div>
                          )
                        }},
                    ]}
                  />
                )}

                {/* Granger shown for ensemble as a secondary panel */}
                {activeMethod === 'ensemble' && grangerRows.length > 0 && (
                  <StatsTable title="Granger Causality (temporal signal)" rows={grangerRows}
                    cols={[
                      { key: 'feature', label: 'Feature', mono: true },
                      { key: 'p_value', label: 'p-value', mono: true, render: v => <span className={v < 0.05 ? 'text-green-400' : 'text-amber-400'}>{v?.toFixed(4)}</span> },
                      { key: 'significant', label: 'Granger causal?', render: v => <Badge color={v ? 'green' : 'faint'}>{v ? 'Yes' : 'No'}</Badge> },
                    ]}
                  />
                )}

                {/* Graph structure info for graph method */}
                {activeMethod === 'graph' && (() => {
                  const g = result?.methods?.graph?.graph ?? result?.graph ?? {}
                  const nodes = g.nodes ?? []
                  const edges = g.edges ?? []
                  const causalEdges = edges.filter(e => e.is_causal)
                  return nodes.length > 0 ? (
                    <div className="bg-sentinel-card border border-sentinel-border rounded-xl p-4 space-y-3">
                      <div className="text-xs font-semibold text-sentinel-text">Causal Graph Structure</div>
                      <div className="flex gap-6 text-xs text-sentinel-faint">
                        <span><span className="text-sentinel-text font-medium">{nodes.length}</span> nodes</span>
                        <span><span className="text-sentinel-text font-medium">{edges.length}</span> total edges</span>
                        <span><span className="text-amber-400 font-medium">{causalEdges.length}</span> Granger causal</span>
                        <span><span className="text-cyan-400 font-medium">{usedPC ? 'PC algorithm' : 'corr DAG'}</span></span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {nodes.slice(0, 10).map(n => (
                          <div key={n.id} className={clsx(
                            'flex items-center gap-2 px-2 py-1.5 rounded-lg border text-xs',
                            n.is_target
                              ? 'bg-red-500/10 border-red-500/20 text-red-400'
                              : causalEdges.some(e => e.source === n.id)
                              ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                              : 'bg-sentinel-hover border-sentinel-border text-sentinel-muted'
                          )}>
                            <span className={clsx('w-2 h-2 rounded-full flex-shrink-0',
                              n.is_target ? 'bg-red-500' : causalEdges.some(e => e.source === n.id) ? 'bg-amber-400' : 'bg-blue-400'
                            )} />
                            {n.id}
                            {n.is_target && <Badge color="red">target</Badge>}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null
                })()}
              </div>
            )}

          </div>
        )}
      </div>

      {/* Chat drawer */}
      {chatOpen && result && (
        <ChatDrawer
          onClose={() => setChatOpen(false)}
          context={chatContext}
          onPendingCharts={(p) => setPendingSuggestion(p)}
          messages={rcaChatMessages}
          setMessages={setRcaChatMessages}
        />
      )}

      {/* Traverse modal */}
      {traverseFeature && result && (
        <TraverseModal feature={traverseFeature} table={result.table} targetCol={result.target_col} onClose={() => setTraverse(null)} />
      )}
    </div>
  )
}
