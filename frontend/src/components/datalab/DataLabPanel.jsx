import React, { useState, useEffect, useRef, useCallback } from 'react'
import clsx from 'clsx'
import {
  Table2, Download, Play, Database,
  Code2, BarChart3, Wand2, Loader,
  AlertTriangle, ChevronDown, ChevronUp, Send,
  RefreshCw, Sparkles, Info, Search,
  Zap, Trash2, FileText, Upload, Copy, X,
  CheckCircle2, Loader2,
} from 'lucide-react'
import { useApp } from '../../context/AppContext'
import {
  getDataLabTables, previewTable, runDataLabSql,
  transformData, autoPlotTable, requestCustomPlot, downloadTable,
  getTableSchema, identifyDataset, querySchema,
  promoteDataset, dropTable, uploadFile,
} from '../../api/client'

// ── Helpers ───────────────────────────────────────────────────────────────────
function _safeStr(v) {
  if (v === null || v === undefined) return <span className="text-sentinel-faint italic text-[10px]">null</span>
  return String(v)
}

function TypeBadge({ dtype = '' }) {
  const d = dtype.toUpperCase()
  if (/INT|FLOAT|DOUBLE|DECIMAL|NUMERIC|REAL/.test(d))
    return <span className="text-[9px] px-1 py-0.5 rounded bg-sentinel-blue/10 text-sentinel-blue border border-sentinel-blue/20">num</span>
  if (/DATE|TIME|STAMP/.test(d))
    return <span className="text-[9px] px-1 py-0.5 rounded bg-sentinel-green/10 text-sentinel-green border border-sentinel-green/20">date</span>
  if (/BOOL/.test(d))
    return <span className="text-[9px] px-1 py-0.5 rounded bg-sentinel-yellow/10 text-sentinel-yellow border border-sentinel-yellow/20">bool</span>
  return <span className="text-[9px] px-1 py-0.5 rounded bg-sentinel-cyan/10 text-sentinel-cyan border border-sentinel-cyan/20">text</span>
}

// ── Data table ────────────────────────────────────────────────────────────────
function DataTable({ columns, rows, dtypes = {}, emptyMsg = 'No data' }) {
  if (!columns?.length) return (
    <div className="flex items-center justify-center py-12 text-sentinel-faint text-sm">{emptyMsg}</div>
  )
  return (
    <div className="overflow-auto flex-1 min-h-0 rounded-xl border border-sentinel-border">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-sentinel-card border-b border-sentinel-border z-10">
          <tr>
            <th className="w-8 px-2 py-2 text-sentinel-faint font-normal text-right">#</th>
            {columns.map(col => (
              <th key={col} className="px-3 py-2 text-left whitespace-nowrap">
                <div className="flex items-center gap-1">
                  <span className="text-sentinel-muted font-semibold">{col}</span>
                  {dtypes[col] && <TypeBadge dtype={dtypes[col]} />}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={clsx('border-b border-sentinel-border/40 hover:bg-sentinel-card/60', i % 2 && 'bg-sentinel-surface/30')}>
              <td className="px-2 py-1.5 text-sentinel-faint text-right text-[10px]">{i + 1}</td>
              {columns.map(col => (
                <td key={col} className="px-3 py-1.5 text-sentinel-text whitespace-nowrap max-w-[200px] truncate">
                  {_safeStr(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Chart iframe ──────────────────────────────────────────────────────────────
function ChartCard({ chart }) {
  return (
    <div className="rounded-xl border border-sentinel-border overflow-hidden bg-sentinel-card">
      <div className="px-3 py-2 border-b border-sentinel-border text-xs font-medium text-sentinel-muted">
        {chart.title}
      </div>
      <iframe
        srcDoc={chart.html}
        title={chart.title}
        className="w-full"
        style={{ height: 340, border: 'none', background: '#111827' }}
        sandbox="allow-scripts"
      />
    </div>
  )
}

// ── Code block ────────────────────────────────────────────────────────────────
function CodeBlock({ code, label = 'Generated Code' }) {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="rounded-lg border border-sentinel-border overflow-hidden">
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-sentinel-card text-xs text-sentinel-faint hover:text-sentinel-muted transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <Code2 className="w-3 h-3 text-sentinel-purple" />
          <span className="text-sentinel-purple font-medium">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); copy() }}
            className="text-[10px] px-2 py-0.5 rounded bg-sentinel-card border border-sentinel-border hover:border-sentinel-purple/40 hover:text-sentinel-purple transition-colors"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </div>
      </button>
      {expanded && (
        <pre className="px-3 py-3 text-[10px] font-mono text-sentinel-cyan bg-black/40 overflow-x-auto whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto">
          {code}
        </pre>
      )}
    </div>
  )
}

// ── Dataset identifier banner ─────────────────────────────────────────────────
function DatasetHint({ table, confidence, reason, ambiguous, candidates, onSelectTable }) {
  if (!table && !ambiguous) return null
  if (ambiguous) {
    return (
      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-sentinel-yellow/10 border border-sentinel-yellow/20">
        <AlertTriangle className="w-3.5 h-3.5 text-sentinel-yellow mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-xs text-sentinel-yellow font-medium">Which dataset?</div>
          <div className="text-[10px] text-sentinel-yellow/80 mt-0.5">{reason}</div>
          <div className="flex gap-1.5 flex-wrap mt-1.5">
            {candidates?.map(t => (
              <button key={t} onClick={() => onSelectTable(t)}
                className="text-[10px] px-2 py-0.5 rounded border border-sentinel-yellow/30 text-sentinel-yellow hover:bg-sentinel-yellow/10 transition-colors">
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-sentinel-cyan/5 border border-sentinel-cyan/20">
      <Search className="w-3 h-3 text-sentinel-cyan flex-shrink-0" />
      <span className="text-[10px] text-sentinel-cyan">
        Targeting <strong>{table}</strong>
        {confidence >= 0.9 ? ' (exact match)' : confidence >= 0.75 ? ' (AI identified)' : ' (best guess)'}
        {reason ? ` — ${reason}` : ''}
      </span>
    </div>
  )
}

// ── TRANSFORM TAB ─────────────────────────────────────────────────────────────
function TransformTab({ activeTable, previewData, allTables, onPromoted }) {
  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [verifierNotes, setVerifierNotes] = useState('')

  // Stack: [{label, code, sql, mode, columns, rows, row_count, dtypes}]
  const [stack, setStack] = useState([])
  const [identifying, setIdentifying] = useState(false)
  const [identified, setIdentified] = useState(null)   // {table, confidence, reason, ambiguous, candidates}
  const [targetTable, setTargetTable] = useState(activeTable)

  const textRef = useRef(null)

  // When active table changes, reset
  useEffect(() => {
    setStack([])
    setError(null)
    setIdentified(null)
    setTargetTable(activeTable)
    setVerifierNotes('')
  }, [activeTable])

  const currentStep = stack.length ? stack[stack.length - 1] : null
  const displayData = currentStep ?? {
    columns: previewData?.columns,
    rows: previewData?.rows,
    dtypes: previewData?.dtypes,
    row_count: previewData?.row_count,
  }

  const currentSql = stack.find(s => s.mode === 'sql' && s.sql)?.sql ?? null
  // All prior pandas codes in order — backend will replay these to rebuild accumulated df
  const priorStepCodes = stack.filter(s => s.mode === 'pandas' && s.code).map(s => s.code)

  // Auto-identify dataset from prompt
  const identifyTarget = async (promptText) => {
    if (!promptText.trim() || allTables.length <= 1) return activeTable
    setIdentifying(true)
    try {
      const res = await identifyDataset(promptText)
      setIdentified(res)
      if (!res.ambiguous && res.table) {
        setTargetTable(res.table)
        return res.table
      }
    } catch (_) {}
    finally { setIdentifying(false) }
    return targetTable
  }

  const apply = async () => {
    if (!prompt.trim()) return
    setLoading(true); setError(null); setVerifierNotes('')

    // Identify target table from prompt context
    const table = await identifyTarget(prompt)
    if (!table) {
      setLoading(false)
      return
    }

    try {
      const res = await transformData({
        table,
        prompt: prompt.trim(),
        current_sql: currentSql ?? undefined,
        prior_step_codes: priorStepCodes,
        use_pandas: true,
      })
      if (!res.success) {
        setError(res.error || 'Transform failed')
        if (res.verifier_notes) setVerifierNotes(res.verifier_notes)
      } else {
        if (res.verifier_notes) setVerifierNotes(res.verifier_notes)
        setStack(prev => [...prev, {
          label: prompt.trim(),
          code: res.code || '',
          sql: res.sql || '',
          mode: res.mode || 'pandas',
          columns: res.columns,
          rows: res.rows,
          row_count: res.row_count,
          dtypes: Object.fromEntries((res.columns || []).map(c => [c, 'object'])),
        }])
        setPrompt('')
      }
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  // Remove ONLY step at index i — keep all other steps intact
  const removeStep = (i) => { setStack(prev => prev.filter((_, idx) => idx !== i)); setError(null); setVerifierNotes('') }
  const reset = () => { setStack([]); setError(null); setIdentified(null); setTargetTable(activeTable); setVerifierNotes('') }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); apply() }
  }

  const EXAMPLES = [
    'Create total spending per customer',
    'Normalize price to 0-1 range',
    'Drop rows where revenue is null',
    'Compute discount percentage from discount_amount and base_amount',
    'One-hot encode the category column',
    'Standardize all numeric columns (z-score)',
    'Bin order_value into low / mid / high groups',
    'Aggregate revenue and quantity by month',
    'Compute rolling 7-day average of sales',
    'Find top 10% customers by total spend',
  ]

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">

      {/* Breadcrumb undo stack — each step has an × to remove it and all after */}
      {stack.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap flex-shrink-0">
          <span className="text-[10px] text-sentinel-faint uppercase tracking-wider">Steps:</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-sentinel-card border border-sentinel-border text-sentinel-muted">
            {targetTable || activeTable} (original)
          </span>
          {stack.map((s, i) => (
            <React.Fragment key={i}>
              <span className="text-sentinel-faint text-[10px]">›</span>
              <span
                className={clsx(
                  'group text-[10px] pl-2 pr-1 py-0.5 rounded border flex items-center gap-1 max-w-[200px]',
                  s.mode === 'pandas'
                    ? 'bg-sentinel-purple/10 border-sentinel-purple/20 text-sentinel-purple'
                    : 'bg-sentinel-blue/10 border-sentinel-blue/20 text-sentinel-blue'
                )}
              >
                <span className="truncate max-w-[140px]" title={s.label}>
                  {s.mode === 'pandas' ? 'py' : 'sql'} · {s.label}
                </span>
                <button
                  onClick={() => removeStep(i)}
                  title={`Remove this step`}
                  className="flex-shrink-0 opacity-50 hover:opacity-100 hover:text-sentinel-red transition-opacity rounded"
                >
                  ×
                </button>
              </span>
            </React.Fragment>
          ))}
          <button onClick={reset} title="Reset to original"
            className="ml-1 p-1 rounded hover:bg-sentinel-card text-sentinel-faint hover:text-sentinel-red transition-colors">
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Dataset identifier */}
      {identified && (
        <DatasetHint
          {...identified}
          onSelectTable={(t) => { setTargetTable(t); setIdentified(null) }}
        />
      )}

      {/* Verifier notes */}
      {verifierNotes && (
        <div className="flex items-start gap-2 p-2 rounded-lg bg-sentinel-yellow/10 border border-sentinel-yellow/20 flex-shrink-0">
          <Zap className="w-3 h-3 text-sentinel-yellow mt-0.5 flex-shrink-0" />
          <span className="text-[10px] text-sentinel-yellow/90">{verifierNotes}</span>
        </div>
      )}

      {/* Code preview (collapsible) */}
      {currentStep?.code && (
        <div className="flex-shrink-0">
          <CodeBlock
            code={currentStep.code}
            label={currentStep.mode === 'pandas' ? 'Python (pandas/numpy) Code' : 'SQL Fallback Code'}
          />
        </div>
      )}

      {/* Prompt input */}
      <div className="flex gap-2 flex-shrink-0">
        <div className="flex-1 relative">
          <Wand2 className="absolute left-3 top-3 w-3.5 h-3.5 text-sentinel-faint pointer-events-none" />
          {identifying && (
            <Loader className="absolute right-3 top-3 w-3 h-3 text-sentinel-cyan animate-spin pointer-events-none" />
          )}
          <textarea
            ref={textRef}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKey}
            rows={2}
            className="w-full pl-8 pr-8 py-2.5 bg-sentinel-card border border-sentinel-border rounded-xl text-sm text-sentinel-text placeholder-sentinel-faint focus:outline-none focus:border-sentinel-purple/50 focus:ring-1 focus:ring-sentinel-purple/30 transition-all resize-none"
            placeholder='Describe a transformation… e.g. "compute discount % from discount_amount/base_amount", "normalize price"  (Enter to apply)'
          />
        </div>
        <button
          onClick={apply}
          disabled={loading || !prompt.trim()}
          className="px-4 rounded-xl bg-sentinel-purple/10 hover:bg-sentinel-purple/20 border border-sentinel-purple/20 text-sentinel-purple font-semibold text-sm flex items-center gap-2 disabled:opacity-40 transition-colors self-stretch"
        >
          {loading ? <Loader className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Apply
        </button>
      </div>

      {/* Example chips */}
      {stack.length === 0 && !loading && (
        <div className="flex gap-1.5 flex-wrap flex-shrink-0">
          {EXAMPLES.map(ex => (
            <button key={ex} onClick={() => setPrompt(ex)}
              className="text-[10px] px-2 py-1 rounded-full border border-sentinel-border text-sentinel-faint hover:border-sentinel-purple/30 hover:text-sentinel-muted transition-colors">
              {ex}
            </button>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-2.5 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20 flex-shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 text-sentinel-red mt-0.5 flex-shrink-0" />
          <span className="text-xs text-sentinel-red leading-relaxed">{error}</span>
        </div>
      )}

      {/* Stats + Copy to Intelligence */}
      {displayData?.row_count != null && (
        <div className="text-[10px] text-sentinel-faint flex-shrink-0 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span>{displayData.row_count?.toLocaleString()} rows · {displayData.columns?.length} columns</span>
            {stack.length > 0 && (
              <span className={clsx('font-medium', currentStep?.mode === 'pandas' ? 'text-sentinel-purple' : 'text-sentinel-blue')}>
                ({currentStep?.mode === 'pandas' ? 'pandas transform' : 'SQL transform'})
              </span>
            )}
          </div>
          {stack.length > 0 && (
            <CopyToIntelligenceBtn
              table={targetTable || activeTable}
              priorStepCodes={priorStepCodes}
              currentSql={currentSql}
              onPromoted={onPromoted}
            />
          )}
        </div>
      )}

      {/* Result table */}
      <DataTable
        columns={displayData?.columns}
        rows={displayData?.rows ?? []}
        dtypes={displayData?.dtypes ?? {}}
        emptyMsg="Apply a transformation to see the result"
      />
    </div>
  )
}

// ── Copy to Intelligence button ───────────────────────────────────────────────
function CopyToIntelligenceBtn({ table, priorStepCodes, currentSql, onPromoted }) {
  const [promoting, setPromoting] = useState(false)
  const [result, setResult] = useState(null) // { success, new_table, error }
  const { setHasModified, addDataset } = useApp()

  const handlePromote = async () => {
    setPromoting(true); setResult(null)
    try {
      const res = await promoteDataset({
        table,
        prior_step_codes: priorStepCodes || [],
        current_sql: currentSql || undefined,
      })
      setResult(res)
      if (res.success) {
        // Enable the version toggle everywhere
        setHasModified(true)
        // Register the modified dataset so QueryInput can find it
        addDataset({
          filename: res.filename || `${res.new_table}.modified`,
          tables: [res.new_table],
          row_count: res.row_count || 0,
        })
        // Notify parent to refresh table list and switch to modified
        if (onPromoted) onPromoted(res.new_table)
      }
    } catch (e) {
      setResult({ success: false, error: e?.response?.data?.detail || e.message })
    } finally {
      setPromoting(false)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handlePromote}
        disabled={promoting}
        className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-sentinel-cyan/10 hover:bg-sentinel-cyan/20 border-2 border-sentinel-cyan/30 text-sentinel-cyan text-[11px] font-bold transition-all disabled:opacity-50"
      >
        {promoting ? <Loader className="w-3 h-3 animate-spin" /> : <Copy className="w-3 h-3" />}
        Copy to Intelligence
      </button>
      {result && (
        <span className={clsx('text-[10px] font-medium', result.success ? 'text-sentinel-green' : 'text-sentinel-red')}>
          {result.success ? `✓ Created ${result.new_table}` : `✕ ${result.error}`}
        </span>
      )}
    </div>
  )
}

// ── SCHEMA TAB ────────────────────────────────────────────────────────────────
const SCHEMA_QUERY_EXAMPLES = [
  'Show null counts', 'Find duplicates', 'Show data types',
  'Describe statistics', 'Unique counts per column', 'Memory usage',
  'Correlation matrix', 'Value counts', 'Find outliers', 'Skewness',
  'Kurtosis', 'Shape / dimensions', 'Numeric columns', 'Categorical columns',
  'Date columns', 'Min and max per column', 'Zero values',
  'Constant columns', 'High cardinality columns', 'Show sample rows',
]

function SchemaTab({ activeTable }) {
  const [schemaData, setSchemaData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')

  // NL query state
  const [nlPrompt, setNlPrompt] = useState('')
  const [nlLoading, setNlLoading] = useState(false)
  const [nlResult, setNlResult] = useState(null)   // {code, result_json, format, mode, error}
  const [nlError, setNlError] = useState(null)
  const [showCode, setShowCode] = useState(false)

  useEffect(() => {
    if (!activeTable) return
    setLoading(true); setSchemaData(null); setError(null); setNlResult(null)
    getTableSchema(activeTable)
      .then(d => setSchemaData(d))
      .catch(e => setError(e?.response?.data?.detail || e.message))
      .finally(() => setLoading(false))
  }, [activeTable])

  const runNlQuery = async () => {
    if (!nlPrompt.trim() || !activeTable) return
    setNlLoading(true); setNlResult(null); setNlError(null)
    try {
      const res = await querySchema(activeTable, nlPrompt.trim())
      if (res.success) setNlResult(res)
      else setNlError(res.error || 'Query failed')
    } catch (e) {
      setNlError(e?.response?.data?.detail || e.message)
    } finally { setNlLoading(false) }
  }

  if (loading) return (
    <div className="flex items-center justify-center py-16 text-sentinel-faint">
      <Loader className="w-5 h-5 animate-spin mr-2" /> Analyzing schema...
    </div>
  )
  if (error) return (
    <div className="flex items-center gap-2 p-3 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20">
      <AlertTriangle className="w-4 h-4 text-sentinel-red" />
      <span className="text-sm text-sentinel-red">{error}</span>
    </div>
  )
  if (!schemaData) return null

  const filtered = schemaData.columns.filter(c =>
    !search || c.name.toLowerCase().includes(search.toLowerCase())
  )
  const nullCols = filtered.filter(c => c.null_count > 0)

  // Parse NL result rows for table display
  let nlRows = [], nlCols = []
  if (nlResult?.result_json) {
    try {
      const parsed = JSON.parse(nlResult.result_json)
      if (Array.isArray(parsed) && parsed.length > 0) {
        nlCols = Object.keys(parsed[0])
        nlRows = parsed
      }
    } catch (_) {}
  }

  return (
    <div className="flex flex-col gap-4 h-full min-h-0 overflow-y-auto pr-1">

      {/* NL Query box */}
      <div className="bg-sentinel-card rounded-xl border border-sentinel-border p-3 flex-shrink-0">
        <div className="text-xs font-semibold text-sentinel-muted mb-2 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-sentinel-cyan" /> Ask about this dataset's schema
        </div>
        <div className="flex gap-2">
          <input
            value={nlPrompt}
            onChange={e => setNlPrompt(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') runNlQuery() }}
            placeholder='e.g. "show null counts", "find outliers", "correlation matrix"'
            className="flex-1 px-3 py-2 bg-sentinel-surface border border-sentinel-border rounded-lg text-sm text-sentinel-text placeholder-sentinel-faint focus:outline-none focus:border-sentinel-cyan/40 transition-all"
          />
          <button
            onClick={runNlQuery}
            disabled={nlLoading || !nlPrompt.trim()}
            className="px-3 py-2 rounded-lg bg-sentinel-cyan/10 hover:bg-sentinel-cyan/20 border border-sentinel-cyan/20 text-sentinel-cyan text-sm font-medium disabled:opacity-40 transition-colors flex items-center gap-1.5"
          >
            {nlLoading ? <Loader className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            Run
          </button>
        </div>
        {/* Example chips */}
        <div className="flex gap-1.5 flex-wrap mt-2">
          {SCHEMA_QUERY_EXAMPLES.map(ex => (
            <button key={ex} onClick={() => { setNlPrompt(ex); }}
              className="text-[10px] px-2 py-0.5 rounded-full border border-sentinel-border text-sentinel-faint hover:border-sentinel-cyan/30 hover:text-sentinel-muted transition-colors">
              {ex}
            </button>
          ))}
        </div>
      </div>

      {/* NL Query result */}
      {nlError && (
        <div className="flex items-start gap-2 p-2.5 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20 flex-shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 text-sentinel-red mt-0.5 flex-shrink-0" />
          <span className="text-xs text-sentinel-red">{nlError}</span>
        </div>
      )}
      {nlResult && (
        <div className="bg-sentinel-card rounded-xl border border-sentinel-border flex-shrink-0 overflow-hidden">
          <div className="flex items-center justify-between px-3 py-2 border-b border-sentinel-border">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-sentinel-text">Result</span>
              <span className={clsx(
                'text-[9px] px-1.5 py-0.5 rounded border font-medium',
                nlResult.mode === 'hardcoded'
                  ? 'bg-sentinel-green/10 border-sentinel-green/20 text-sentinel-green'
                  : 'bg-sentinel-purple/10 border-sentinel-purple/20 text-sentinel-purple'
              )}>
                {nlResult.mode === 'hardcoded' ? 'instant' : 'AI generated'}
              </span>
            </div>
            {nlResult.code && (
              <button onClick={() => setShowCode(v => !v)}
                className="text-[10px] text-sentinel-faint hover:text-sentinel-muted flex items-center gap-1 transition-colors">
                <Code2 className="w-3 h-3" />
                {showCode ? 'Hide code' : 'Show code'}
              </button>
            )}
          </div>
          {showCode && nlResult.code && (
            <pre className="px-3 py-2 text-[10px] font-mono text-sentinel-cyan/90 bg-[#0A0E1A] border-b border-sentinel-border overflow-x-auto">
              {nlResult.code}
            </pre>
          )}
          {nlRows.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-sentinel-surface border-b border-sentinel-border">
                  <tr>
                    {nlCols.map(c => (
                      <th key={c} className="px-3 py-2 text-left text-sentinel-faint font-medium whitespace-nowrap">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {nlRows.slice(0, 50).map((row, i) => (
                    <tr key={i} className={clsx('border-b border-sentinel-border/40', i % 2 && 'bg-sentinel-surface/20')}>
                      {nlCols.map(c => (
                        <td key={c} className="px-3 py-1.5 text-sentinel-muted whitespace-nowrap max-w-[200px] truncate">
                          {row[c] == null ? <span className="text-sentinel-faint italic text-[10px]">null</span> : String(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <pre className="px-3 py-2 text-[10px] font-mono text-sentinel-muted overflow-x-auto max-h-48">
              {nlResult.result_json}
            </pre>
          )}
        </div>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3 flex-shrink-0">
        {[
          { label: 'Rows', value: schemaData.row_count.toLocaleString(), color: 'text-sentinel-text' },
          { label: 'Columns', value: schemaData.col_count, color: 'text-sentinel-blue' },
          { label: 'Nulls', value: nullCols.length + ' cols', color: nullCols.length > 0 ? 'text-sentinel-yellow' : 'text-sentinel-green' },
          { label: 'Memory', value: schemaData.memory_mb.toFixed(1) + ' MB', color: 'text-sentinel-cyan' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-sentinel-card rounded-xl p-3 border border-sentinel-border">
            <div className="text-[10px] text-sentinel-faint uppercase tracking-wider">{label}</div>
            <div className={clsx('text-lg font-bold mt-1', color)}>{value}</div>
          </div>
        ))}
      </div>

      {/* Search */}
      <div className="relative flex-shrink-0">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-sentinel-faint" />
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search columns..."
          className="w-full pl-8 pr-3 py-2 bg-sentinel-card border border-sentinel-border rounded-xl text-sm text-sentinel-text placeholder-sentinel-faint focus:outline-none focus:border-sentinel-blue/40 transition-all"
        />
      </div>

      {/* Column detail table */}
      <div className="rounded-xl border border-sentinel-border overflow-hidden flex-shrink-0">
        <table className="w-full text-xs">
          <thead className="bg-sentinel-card border-b border-sentinel-border">
            <tr>
              <th className="px-3 py-2.5 text-left text-sentinel-faint font-medium">Column</th>
              <th className="px-3 py-2.5 text-left text-sentinel-faint font-medium">Type</th>
              <th className="px-3 py-2.5 text-right text-sentinel-faint font-medium">Unique</th>
              <th className="px-3 py-2.5 text-right text-sentinel-faint font-medium">Nulls</th>
              <th className="px-3 py-2.5 text-right text-sentinel-faint font-medium">Min</th>
              <th className="px-3 py-2.5 text-right text-sentinel-faint font-medium">Max</th>
              <th className="px-3 py-2.5 text-right text-sentinel-faint font-medium">Mean</th>
              <th className="px-3 py-2.5 text-left text-sentinel-faint font-medium">Samples</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((col, i) => (
              <tr key={col.name} className={clsx(
                'border-b border-sentinel-border/40 hover:bg-sentinel-card/60',
                i % 2 && 'bg-sentinel-surface/20'
              )}>
                <td className="px-3 py-2 font-medium text-sentinel-text">{col.name}</td>
                <td className="px-3 py-2"><TypeBadge dtype={col.dtype} /></td>
                <td className="px-3 py-2 text-right text-sentinel-muted">{col.unique_count.toLocaleString()}</td>
                <td className={clsx(
                  'px-3 py-2 text-right font-medium',
                  col.null_count > 0 ? 'text-sentinel-yellow' : 'text-sentinel-green'
                )}>
                  {col.null_count > 0 ? `${col.null_pct}%` : '—'}
                </td>
                <td className="px-3 py-2 text-right text-sentinel-faint font-mono text-[10px]">
                  {col.min_val != null ? String(col.min_val) : '—'}
                </td>
                <td className="px-3 py-2 text-right text-sentinel-faint font-mono text-[10px]">
                  {col.max_val != null ? String(col.max_val) : '—'}
                </td>
                <td className="px-3 py-2 text-right text-sentinel-faint font-mono text-[10px]">
                  {col.mean_val != null ? String(col.mean_val) : '—'}
                </td>
                <td className="px-3 py-2 max-w-[200px]">
                  <div className="flex gap-1 flex-wrap">
                    {col.sample_values?.slice(0, 4).map((v, idx) => (
                      <span key={idx} className="text-[9px] px-1 py-0.5 rounded bg-sentinel-surface border border-sentinel-border text-sentinel-muted truncate max-w-[80px]">
                        {v == null ? 'null' : String(v)}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Null columns warning */}
      {nullCols.length > 0 && (
        <div className="p-3 rounded-lg bg-sentinel-yellow/10 border border-sentinel-yellow/20 flex-shrink-0">
          <div className="text-xs font-medium text-sentinel-yellow mb-1.5">Columns with missing values</div>
          <div className="flex gap-2 flex-wrap">
            {nullCols.map(c => (
              <span key={c.name} className="text-[10px] px-2 py-0.5 rounded bg-sentinel-yellow/10 border border-sentinel-yellow/20 text-sentinel-yellow">
                {c.name} ({c.null_pct}% null)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── PLOTS TAB ─────────────────────────────────────────────────────────────────
function PlotsTab({ activeTable, currentSql, allTables = [] }) {
  const [autoCharts, setAutoCharts] = useState([])
  const [extraCharts, setExtraCharts] = useState([])
  const [autoLoading, setAutoLoading] = useState(false)
  const [plotPrompt, setPlotPrompt] = useState('')
  const [plotLoading, setPlotLoading] = useState(false)
  const [plotError, setPlotError] = useState(null)
  const [plotCode, setPlotCode] = useState('')

  // Version toggle: 'original' | 'modified' — scoped to plots only
  const baseTable = activeTable?.replace(/_modified$/, '') || ''
  const modifiedTable = `${baseTable}_modified`
  const hasModifiedVersion = allTables.includes(modifiedTable) || activeTable?.endsWith('_modified')
  const [plotVersion, setPlotVersion] = useState('original')

  // Resolve which table to use for plots based on toggle
  const plotTable = (plotVersion === 'modified' && hasModifiedVersion)
    ? modifiedTable
    : (activeTable?.endsWith('_modified') ? baseTable : activeTable)

  // Reset version when active table changes
  useEffect(() => { setPlotVersion('original') }, [activeTable])

  useEffect(() => {
    if (!plotTable) return
    setAutoCharts([]); setAutoLoading(true)
    autoPlotTable(plotTable, currentSql ?? '')
      .then(r => setAutoCharts(r.charts ?? []))
      .catch(() => setAutoCharts([]))
      .finally(() => setAutoLoading(false))
  }, [plotTable, currentSql])

  const requestPlot = async () => {
    if (!plotPrompt.trim() || !plotTable) return
    setPlotLoading(true); setPlotError(null); setPlotCode('')
    try {
      const res = await requestCustomPlot({
        table: plotTable,
        prompt: plotPrompt.trim(),
        current_sql: currentSql ?? undefined,
      })
      if (!res.success) {
        setPlotError(res.error || 'No chart generated')
        if (res.code) setPlotCode(res.code)
      } else {
        setExtraCharts(prev => [...res.charts, ...prev])
        if (res.code) setPlotCode(res.code)
        setPlotPrompt('')
      }
    } catch (e) {
      setPlotError(e?.response?.data?.detail || e.message)
    } finally {
      setPlotLoading(false)
    }
  }

  const CHART_EXAMPLES = [
    'Show revenue distribution as a histogram',
    'Scatter plot of price vs quantity',
    'Bar chart of top 10 cities by count',
    'Box plot of order value by category',
    'Pie chart of payment methods',
    'Heatmap of numeric correlations',
    'Line chart of revenue over time',
    'Violin plot of price distribution by segment',
  ]

  return (
    <div className="flex flex-col gap-4 h-full min-h-0 overflow-y-auto pr-1">

      {/* Original / Modified toggle for plots */}
      {hasModifiedVersion && (
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-[10px] font-bold text-sentinel-faint uppercase tracking-wider">Plot data:</span>
          <div className="flex rounded-lg overflow-hidden border-2 border-sentinel-border">
            {['original', 'modified'].map(v => (
              <button
                key={v}
                onClick={() => setPlotVersion(v)}
                className={clsx(
                  'px-3 py-1.5 text-xs font-bold transition-all capitalize',
                  plotVersion === v
                    ? v === 'modified'
                      ? 'bg-sentinel-cyan/20 text-sentinel-cyan border-sentinel-cyan/40'
                      : 'bg-sentinel-blue/20 text-sentinel-blue border-sentinel-blue/40'
                    : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted'
                )}
              >
                {v === 'original' ? '⬡ Original' : '⬢ Modified'}
              </button>
            ))}
          </div>
          <span className="text-[9px] text-sentinel-faint">
            Using: <strong className={plotVersion === 'modified' ? 'text-sentinel-cyan' : 'text-sentinel-blue'}>{plotTable}</strong>
          </span>
        </div>
      )}

      <div className="flex-shrink-0">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Sparkles className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-sentinel-faint pointer-events-none" />
            <input
              value={plotPrompt}
              onChange={e => setPlotPrompt(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && requestPlot()}
              className="w-full pl-8 pr-3 py-2.5 bg-sentinel-card border border-sentinel-border rounded-xl text-sm text-sentinel-text placeholder-sentinel-faint focus:outline-none focus:border-sentinel-purple/50 focus:ring-1 focus:ring-sentinel-purple/30 transition-all"
              placeholder='Request a chart… e.g. "scatter of price vs sales colored by category"'
            />
          </div>
          <button
            onClick={requestPlot}
            disabled={plotLoading || !plotPrompt.trim()}
            className="px-4 rounded-xl bg-sentinel-purple/10 hover:bg-sentinel-purple/20 border border-sentinel-purple/20 text-sentinel-purple font-semibold text-sm flex items-center gap-2 disabled:opacity-40 transition-colors"
          >
            {plotLoading ? <Loader className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
            Plot
          </button>
        </div>
        {plotError && (
          <div className="mt-2 flex items-start gap-2 p-2 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20">
            <AlertTriangle className="w-3 h-3 text-sentinel-red mt-0.5 flex-shrink-0" />
            <span className="text-xs text-sentinel-red">{plotError}</span>
          </div>
        )}
        {plotCode && !plotLoading && (
          <div className="mt-2">
            <CodeBlock code={plotCode} label="Generated Plot Code" />
          </div>
        )}
        <div className="mt-2 flex gap-1.5 flex-wrap">
          {CHART_EXAMPLES.map(ex => (
            <button key={ex} onClick={() => setPlotPrompt(ex)}
              className="text-[10px] px-2 py-1 rounded-full border border-sentinel-border text-sentinel-faint hover:border-sentinel-purple/30 hover:text-sentinel-muted transition-colors">
              {ex}
            </button>
          ))}
        </div>
      </div>

      {extraCharts.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold text-sentinel-faint uppercase tracking-wider mb-2">Requested Charts</div>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {extraCharts.map((c, i) => <ChartCard key={i} chart={c} />)}
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="text-[10px] font-semibold text-sentinel-faint uppercase tracking-wider">Auto-Generated</div>
          {autoLoading && <Loader className="w-3 h-3 text-sentinel-faint animate-spin" />}
          {!autoLoading && (
            <button onClick={() => {
              setAutoCharts([]); setAutoLoading(true)
              autoPlotTable(plotTable, currentSql ?? '').then(r => setAutoCharts(r.charts ?? [])).finally(() => setAutoLoading(false))
            }} className="p-0.5 rounded text-sentinel-faint hover:text-sentinel-muted transition-colors">
              <RefreshCw className="w-3 h-3" />
            </button>
          )}
        </div>
        {autoLoading ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-64 rounded-xl border border-sentinel-border bg-sentinel-card animate-pulse" />
            ))}
          </div>
        ) : autoCharts.length === 0 ? (
          <div className="flex items-center justify-center py-10 text-sentinel-faint text-sm">
            No charts generated — select a table with numeric data.
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {autoCharts.map((c, i) => <ChartCard key={i} chart={c} />)}
          </div>
        )}
      </div>
    </div>
  )
}

// ── PREVIEW TAB ───────────────────────────────────────────────────────────────
function PreviewTab({ data, loading }) {
  if (loading) return (
    <div className="flex items-center justify-center py-16 text-sentinel-faint">
      <Loader className="w-5 h-5 animate-spin mr-2" /> Loading...
    </div>
  )
  if (!data) return (
    <div className="flex items-center justify-center py-16 text-sentinel-faint text-sm">
      Select a table to preview
    </div>
  )

  const { columns, rows, row_count, dtypes, numeric_summary } = data
  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <div className="flex items-center gap-3 flex-wrap text-xs text-sentinel-faint flex-shrink-0">
        <span><span className="text-sentinel-text font-semibold">{row_count?.toLocaleString()}</span> rows</span>
        <span><span className="text-sentinel-text font-semibold">{columns?.length}</span> columns</span>
        {Object.keys(numeric_summary ?? {}).length > 0 && (
          <span>{Object.keys(numeric_summary).length} numeric cols</span>
        )}
      </div>
      <div className="flex gap-1.5 flex-wrap flex-shrink-0">
        {columns?.map(col => (
          <div key={col} className="flex items-center gap-1 px-2 py-0.5 rounded bg-sentinel-card border border-sentinel-border">
            <span className="text-[11px] text-sentinel-muted">{col}</span>
            <TypeBadge dtype={dtypes?.[col] ?? ''} />
          </div>
        ))}
      </div>
      <DataTable columns={columns} rows={rows ?? []} dtypes={dtypes ?? {}} />
    </div>
  )
}

// ── SQL EDITOR ────────────────────────────────────────────────────────────────
function SqlEditorTab({ activeTable }) {
  const [sql, setSql] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (activeTable) setSql(`SELECT * FROM ${activeTable} LIMIT 100`)
  }, [activeTable])

  const run = async () => {
    if (!sql.trim()) return
    setLoading(true); setError(null)
    try {
      const res = await runDataLabSql(sql)
      if (res.error) setError(res.error)
      else setResult(res)
    } catch (e) { setError(e?.response?.data?.detail || e.message) }
    finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <div className="flex gap-2 flex-shrink-0">
        <textarea
          value={sql}
          onChange={e => setSql(e.target.value)}
          onKeyDown={e => { if (e.ctrlKey && e.key === 'Enter') run() }}
          className="flex-1 bg-sentinel-card border border-sentinel-border rounded-xl px-3 py-2 text-xs text-sentinel-text font-mono resize-none focus:outline-none focus:border-sentinel-blue/40 h-24"
          placeholder="SELECT * FROM your_table LIMIT 100  (Ctrl+Enter to run)"
        />
        <button onClick={run} disabled={loading}
          className="px-3 rounded-xl bg-sentinel-blue/10 hover:bg-sentinel-blue/20 border border-sentinel-blue/20 text-sentinel-blue text-xs font-semibold flex items-center gap-1.5 disabled:opacity-50 self-start py-2">
          {loading ? <Loader className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
          Run
        </button>
      </div>
      {error && (
        <div className="flex items-start gap-2 p-2 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20 flex-shrink-0">
          <AlertTriangle className="w-3 h-3 text-sentinel-red mt-0.5" />
          <span className="text-xs text-sentinel-red">{error}</span>
        </div>
      )}
      {result && (
        <>
          <div className="text-[10px] text-sentinel-faint flex-shrink-0">{result.row_count} rows returned</div>
          <DataTable columns={result.columns} rows={result.rows ?? []} />
        </>
      )}
    </div>
  )
}

// ── Re-Upload Modal (shown when all datasets deleted) ─────────────────────────
function ReUploadModal({ onClose, onUploaded }) {
  const { addDataset, setDataSource, setUploadedFile } = useApp()
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const fileInputRef = useRef(null)

  const doUpload = async (file) => {
    setUploading(true); setUploadError(null); setUploadResult(null)
    try {
      const result = await uploadFile(file, () => {})
      setUploadResult(result)
      setUploadedFile(file)
      setDataSource(file.name.replace(/\.[^.]+$/, ''))
      addDataset({
        filename: file.name,
        tables: result.tables ?? [],
        row_count: result.row_count ?? 0,
        date_min: result.date_min,
        date_max: result.date_max,
      })
      setTimeout(() => onUploaded(result), 600)
    } catch (err) {
      setUploadError(err?.response?.data?.detail || 'Upload failed')
    } finally { setUploading(false) }
  }

  const handleFile = (e) => {
    const file = e.target.files?.[0]
    if (file) doUpload(file)
    e.target.value = ''
  }
  const handleDrop = (e) => {
    e.preventDefault(); setIsDragging(false)
    const files = Array.from(e.dataTransfer.files ?? [])
    const allowed = ['.csv', '.xlsx', '.xls', '.parquet', '.db', '.sqlite', '.sqlite3']
    const validFile = files.find(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)))
    if (validFile) doUpload(validFile)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg card p-8 shadow-sentinel relative animate-fade-in">
        <button onClick={onClose} className="absolute top-3 right-3 p-1 text-sentinel-faint hover:text-sentinel-text transition-colors rounded">
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 mb-4">
          <Database className="w-5 h-5 text-sentinel-cyan" />
          <h2 className="text-lg font-bold text-sentinel-text">Upload Dataset</h2>
        </div>
        <p className="text-sm text-sentinel-muted mb-5">
          All datasets have been removed. Upload a new file to continue working in DataLab.
        </p>

        {uploadResult && (
          <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-sentinel-green/10 border border-sentinel-green/20">
            <CheckCircle2 className="w-4 h-4 text-sentinel-green" />
            <span className="text-sm text-sentinel-green">
              Uploaded! {uploadResult.row_count?.toLocaleString()} rows · {uploadResult.tables?.length} table(s)
            </span>
          </div>
        )}
        {uploadError && (
          <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-sentinel-red/10 border border-sentinel-red/20">
            <AlertTriangle className="w-4 h-4 text-sentinel-red" />
            <span className="text-sm text-sentinel-red">{uploadError}</span>
          </div>
        )}

        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={clsx(
            'flex flex-col items-center gap-3 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 group',
            isDragging
              ? 'border-sentinel-blue bg-sentinel-blue/5 scale-[1.01]'
              : 'border-sentinel-border hover:border-sentinel-blue/50 hover:bg-sentinel-card/50'
          )}
        >
          {uploading ? (
            <Loader2 className="w-8 h-8 text-sentinel-blue animate-spin" />
          ) : (
            <Upload className={clsx('w-8 h-8 transition-colors', isDragging ? 'text-sentinel-blue' : 'text-sentinel-faint group-hover:text-sentinel-blue')} />
          )}
          <div className="text-sm text-sentinel-muted group-hover:text-sentinel-text text-center transition-colors">
            {uploading ? 'Uploading...' : <>Drop your dataset here or <span className="text-sentinel-blue font-medium">browse</span></>}
          </div>
          <div className="text-xs text-sentinel-faint">
            CSV, Excel, Parquet, or SQLite (up to 500 MB)
          </div>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.xls,.parquet,.db,.sqlite,.sqlite3"
            onChange={handleFile}
          />
        </div>
      </div>
    </div>
  )
}

// ── MAIN DataLabPage ──────────────────────────────────────────────────────────
const TABS = [
  { id: 'preview',   label: 'Preview',   icon: Table2 },
  { id: 'schema',    label: 'Schema',    icon: Info },
  { id: 'transform', label: 'Transform', icon: Wand2 },
  { id: 'plots',     label: 'Plots',     icon: BarChart3 },
  { id: 'sql',       label: 'SQL',       icon: Code2 },
]

export default function DataLabPage() {
  const { dataLabTable, setDataLabTable, datasets, removeDataset, addDataset, setDataSource, setUploadedFile } = useApp()

  const [tables, setTables]           = useState([])
  const [activeTable, setActiveTable] = useState(null)
  const [previewData, setPreviewData] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [activeTab, setActiveTab]     = useState('preview')
  const [currentTransformSql, setCurrentTransformSql] = useState(null)
  const [loadingTables, setLoadingTables] = useState(true)
  const [showReUpload, setShowReUpload] = useState(false)

  // Group tables by dataset for display
  const datasetMap = React.useMemo(() => {
    const map = {}
    datasets.forEach(ds => {
      ds.tables?.forEach(t => { map[t] = ds.filename })
    })
    return map
  }, [datasets])

  const loadTables = useCallback(async () => {
    setLoadingTables(true)
    try {
      const t = await getDataLabTables()
      setTables(t)
      const init = dataLabTable ?? t[0] ?? null
      if (init && (!activeTable || !t.includes(activeTable))) {
        setActiveTable(init)
      }
    } catch (_) { setTables([]) }
    finally { setLoadingTables(false) }
  }, [dataLabTable, activeTable])

  useEffect(() => { loadTables() }, [])

  useEffect(() => {
    if (dataLabTable && tables.includes(dataLabTable)) {
      setActiveTable(dataLabTable)
      setDataLabTable(null)
    }
  }, [dataLabTable, tables])

  useEffect(() => {
    if (!activeTable) return
    setPreviewLoading(true); setPreviewData(null); setCurrentTransformSql(null)
    previewTable(activeTable, 200)
      .then(d => setPreviewData(d))
      .catch(() => setPreviewData(null))
      .finally(() => setPreviewLoading(false))
  }, [activeTable])

  const handleRemoveDataset = async (filename) => {
    if (!window.confirm(`Remove "${filename}" and drop all its tables? This cannot be undone.`)) return
    const res = await removeDataset(filename)
    if (res.success) {
      await loadTables()
      if (activeTable && datasetMap[activeTable] === filename) {
        setActiveTable(null)
      }
      // If no datasets remain, show re-upload modal
      const remaining = datasets.filter(d => d.filename !== filename)
      if (remaining.length === 0) {
        setShowReUpload(true)
      }
    }
  }

  const handleDropTable = async (table) => {
    if (!window.confirm(`Drop table "${table}"? This cannot be undone.`)) return
    try {
      await dropTable(table)
      await loadTables()
      if (activeTable === table) setActiveTable(null)
      // Check if all tables are gone
      const remaining = tables.filter(t => t !== table)
      if (remaining.length === 0) {
        setShowReUpload(true)
      }
    } catch (e) {
      console.error('Drop table failed:', e)
    }
  }

  // Called when "Copy to Intelligence" succeeds — refresh tables and switch to modified
  const handlePromoted = async (newTable) => {
    await loadTables()
    // Switch to the modified table so user sees it immediately
    if (newTable) setActiveTable(newTable)
  }

  // Detect if current table has a _modified version
  const baseTable = activeTable?.replace(/_modified$/, '') || ''
  const modifiedTableName = `${baseTable}_modified`
  const hasModified = tables.includes(modifiedTableName)
  const isOnModified = activeTable?.endsWith('_modified')

  const handleToggleVersion = (version) => {
    if (version === 'original') {
      setActiveTable(baseTable)
    } else if (version === 'modified' && hasModified) {
      setActiveTable(modifiedTableName)
    }
  }

  return (
    <div className="flex flex-col h-full bg-sentinel-surface min-h-0">

      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-sentinel-border bg-sentinel-card flex-shrink-0">
        <div className="flex items-center gap-3">
          <Database className="w-4 h-4 text-sentinel-cyan" />
          <div>
            <div className="text-sm font-bold text-sentinel-text">DataLab</div>
            <div className="text-xs text-sentinel-faint">Explore · Schema · Transform (pandas/numpy) · Plot · Query</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {activeTable && (
            <a href={downloadTable(activeTable)} download
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sentinel-green/10 hover:bg-sentinel-green/20 border border-sentinel-green/20 text-sentinel-green text-xs font-semibold transition-colors">
              <Download className="w-3 h-3" /> Export CSV
            </a>
          )}
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Table/Dataset list */}
        <div className="w-52 border-r border-sentinel-border flex flex-col flex-shrink-0 bg-sentinel-card">
          <div className="px-3 py-2.5 border-b border-sentinel-border flex items-center justify-between">
            <span className="text-[10px] font-semibold text-sentinel-faint uppercase tracking-wider">
              Tables {tables.length > 0 ? `(${tables.length})` : ''}
            </span>
            <button onClick={loadTables} title="Refresh tables"
              className="p-0.5 rounded text-sentinel-faint hover:text-sentinel-muted transition-colors">
              {loadingTables ? <Loader className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            </button>
          </div>

          <div className="flex-1 overflow-y-auto py-1">
            {loadingTables ? (
              <div className="px-3 py-4 text-xs text-sentinel-faint text-center">
                <Loader className="w-4 h-4 animate-spin mx-auto" />
              </div>
            ) : tables.length === 0 ? (
              <div className="px-3 py-4 text-xs text-sentinel-faint text-center">No tables.<br />Upload a dataset.</div>
            ) : (
              // Group by dataset
              datasets.length > 0 ? (
                <>
                  {datasets.map(ds => {
                    const dsTables = (ds.tables || []).filter(t => tables.includes(t))
                    if (!dsTables.length) return null
                    return (
                      <div key={ds.filename} className="mb-2">
                        <div className="flex items-center justify-between px-2 py-1 group">
                          <div className="flex items-center gap-1 min-w-0">
                            <FileText className="w-2.5 h-2.5 text-sentinel-faint flex-shrink-0" />
                            <span className="text-[9px] font-semibold text-sentinel-faint uppercase tracking-wider truncate" title={ds.filename}>
                              {ds.filename}
                            </span>
                          </div>
                          <button
                            onClick={() => handleRemoveDataset(ds.filename)}
                            title="Remove dataset"
                            className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-sentinel-faint hover:text-sentinel-red transition-all"
                          >
                            <Trash2 className="w-2.5 h-2.5" />
                          </button>
                        </div>
                        {dsTables.map(t => (
                          <div key={t} className="group flex items-center">
                            <button onClick={() => setActiveTable(t)}
                              className={clsx(
                                'flex-1 flex items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors',
                                activeTable === t
                                  ? 'bg-sentinel-blue/10 text-sentinel-blue border-r-2 border-sentinel-blue'
                                  : 'text-sentinel-muted hover:bg-sentinel-surface hover:text-sentinel-text'
                              )}>
                              <Table2 className="w-3 h-3 flex-shrink-0" />
                              <span className="truncate font-medium">{t}</span>
                              {t.endsWith('_modified') && (
                                <span className="text-[8px] px-1 py-0.5 rounded bg-sentinel-purple/20 text-sentinel-purple flex-shrink-0">MOD</span>
                              )}
                            </button>
                            <button
                              onClick={() => handleDropTable(t)}
                              title={`Drop table ${t}`}
                              className="opacity-0 group-hover:opacity-100 p-1 mr-1 rounded text-sentinel-faint hover:text-sentinel-red transition-all"
                            >
                              <Trash2 className="w-2.5 h-2.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )
                  })}
                  {/* Tables not in any dataset */}
                  {tables.filter(t => !datasetMap[t]).map(t => (
                    <button key={t} onClick={() => setActiveTable(t)}
                      className={clsx(
                        'w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors',
                        activeTable === t
                          ? 'bg-sentinel-blue/10 text-sentinel-blue border-r-2 border-sentinel-blue'
                          : 'text-sentinel-muted hover:bg-sentinel-surface hover:text-sentinel-text'
                      )}>
                      <Table2 className="w-3 h-3 flex-shrink-0" />
                      <span className="truncate font-medium">{t}</span>
                    </button>
                  ))}
                </>
              ) : (
                tables.map(t => (
                  <button key={t} onClick={() => setActiveTable(t)}
                    className={clsx(
                      'w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors',
                      activeTable === t
                        ? 'bg-sentinel-blue/10 text-sentinel-blue border-r-2 border-sentinel-blue'
                        : 'text-sentinel-muted hover:bg-sentinel-surface hover:text-sentinel-text'
                    )}>
                    <Table2 className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate font-medium">{t}</span>
                  </button>
                ))
              )
            )}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 p-5 gap-4">
          {!activeTable ? (
            <div className="flex-1 flex flex-col items-center justify-center text-sentinel-faint">
              <Database className="w-12 h-12 mb-4 opacity-30" />
              <div className="text-lg font-medium mb-2">Select a table</div>
              <div className="text-sm">Choose a table from the left panel to start exploring</div>
            </div>
          ) : (
            <>
              {/* Version dropdown (bold-bordered) */}
              {/* Version Toggle (Original / Modified) — styled pill */}
              <div className="flex items-center gap-3 flex-shrink-0">
                <div className={clsx(
                  'flex rounded-lg overflow-hidden border-2 transition-all duration-300',
                  isOnModified && hasModified
                    ? 'border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
                    : 'border-indigo-500/40 shadow-[0_0_8px_rgba(99,102,241,0.1)]',
                )}>
                  <button
                    onClick={() => handleToggleVersion('original')}
                    className={clsx(
                      'px-3.5 py-1.5 text-xs font-bold transition-all duration-200',
                      !isOnModified
                        ? 'bg-indigo-500/20 text-indigo-300'
                        : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover cursor-pointer',
                    )}
                  >
                    ⬡ Original
                  </button>
                  <button
                    onClick={() => handleToggleVersion('modified')}
                    disabled={!hasModified}
                    className={clsx(
                      'px-3.5 py-1.5 text-xs font-bold transition-all duration-200',
                      isOnModified
                        ? 'bg-cyan-500/20 text-cyan-400'
                        : !hasModified
                          ? 'bg-sentinel-card text-sentinel-faint/40 cursor-not-allowed'
                          : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover cursor-pointer',
                    )}
                  >
                    ⬢ Modified
                  </button>
                </div>
                {!hasModified && (
                  <span className="text-[9px] text-sentinel-faint/50 italic">Transform → Copy to Intelligence to enable</span>
                )}
              </div>

              {/* Table title + tabs */}
              <div className="flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-2">
                  <Table2 className="w-4 h-4 text-sentinel-cyan" />
                  <span className="font-semibold text-sentinel-text">{activeTable}</span>
                  {activeTable?.endsWith('_modified') && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-sentinel-purple/20 border border-sentinel-purple/30 text-sentinel-purple font-bold">MODIFIED</span>
                  )}
                  {datasetMap[activeTable] && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-sentinel-surface border border-sentinel-border text-sentinel-faint">
                      {datasetMap[activeTable]}
                    </span>
                  )}
                  {previewData && (
                    <span className="text-xs text-sentinel-faint">
                      {previewData.row_count?.toLocaleString()} rows · {previewData.columns?.length} cols
                    </span>
                  )}
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-1 bg-sentinel-card rounded-xl p-1 border border-sentinel-border self-start flex-shrink-0">
                {TABS.map(tab => {
                  const Icon = tab.icon
                  return (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                      className={clsx(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                        activeTab === tab.id
                          ? 'bg-sentinel-surface text-sentinel-text shadow-sm'
                          : 'text-sentinel-faint hover:text-sentinel-muted'
                      )}>
                      <Icon className="w-3 h-3" />
                      {tab.label}
                      {tab.id === 'transform' && <span className="text-[8px] px-1 py-0.5 rounded bg-sentinel-purple/20 text-sentinel-purple">AI</span>}
                      {tab.id === 'plots' && <span className="text-[8px] px-1 py-0.5 rounded bg-sentinel-cyan/20 text-sentinel-cyan">AUTO</span>}
                      {tab.id === 'schema' && <span className="text-[8px] px-1 py-0.5 rounded bg-sentinel-green/20 text-sentinel-green">INFO</span>}
                    </button>
                  )
                })}
              </div>

              {/* Tab content */}
              <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
                {activeTab === 'preview' && (
                  <PreviewTab data={previewData} loading={previewLoading} />
                )}
                {activeTab === 'schema' && (
                  <SchemaTab activeTable={activeTable} />
                )}
                {activeTab === 'transform' && (
                  <TransformTab
                    activeTable={activeTable}
                    previewData={previewData}
                    allTables={tables}
                    onPromoted={handlePromoted}
                  />
                )}
                {activeTab === 'plots' && (
                  <PlotsTab activeTable={activeTable} currentSql={currentTransformSql} allTables={tables} />
                )}
                {activeTab === 'sql' && (
                  <SqlEditorTab activeTable={activeTable} />
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Re-upload modal */}
      {showReUpload && (
        <ReUploadModal
          onClose={() => setShowReUpload(false)}
          onUploaded={async () => {
            setShowReUpload(false)
            await loadTables()
          }}
        />
      )}
    </div>
  )
}
