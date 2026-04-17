import React, { useEffect, useState } from 'react'
import { X, Database, GitBranch, Code2, RefreshCw, Loader2 } from 'lucide-react'
import { getMemoryLayer } from '../../api/client'
import clsx from 'clsx'

const LAYER_CONFIG = {
  l2: { label: 'L2 Episodic Memory', icon: Database, color: 'text-sentinel-blue', desc: 'Semantically indexed past query cache using BGE-Large embeddings' },
  l3: { label: 'L3 Causal Graph', icon: GitBranch, color: 'text-sentinel-purple', desc: 'NetworkX causal knowledge graph with business rules and metric relationships' },
  l4: { label: 'L4 Procedural Memory', icon: Code2, color: 'text-sentinel-cyan', desc: 'Verified SQL template patterns for few-shot prompting' },
}

export default function MemoryModal({ layer, onClose, datasets = [] }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeSubTab, setActiveSubTab] = useState(0)
  const [filterDataset, setFilterDataset] = useState('')  // '' = all

  const config = LAYER_CONFIG[layer] || LAYER_CONFIG.l2
  const Icon = config.icon

  const load = async (dsFilter = filterDataset) => {
    setLoading(true)
    setError(null)
    try {
      // Pass dataset filter for L2 layer only
      const ds = (layer === 'l2' && dsFilter) ? dsFilter : null
      const res = await getMemoryLayer(layer, ds)
      setData(res)
    } catch (e) {
      setError('Failed to load memory layer')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load('') }, [layer])

  // Prevent body scroll
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-xl bg-sentinel-surface border-l border-sentinel-border flex flex-col h-full shadow-2xl animate-slide-in overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-sentinel-border flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className={clsx('w-9 h-9 rounded-xl flex items-center justify-center',
              layer === 'l2' ? 'bg-sentinel-blue/10 border border-sentinel-blue/20' :
                layer === 'l3' ? 'bg-sentinel-purple/10 border border-sentinel-purple/20' :
                  'bg-sentinel-cyan/10 border border-sentinel-cyan/20'
            )}>
              <Icon className={clsx('w-5 h-5', config.color)} />
            </div>
            <div>
              <div className="text-sm font-semibold text-sentinel-text">{config.label}</div>
              <div className="text-xs text-sentinel-faint">{config.desc}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => load(filterDataset)} className="btn-ghost p-2" title="Refresh">
              <RefreshCw className="w-4 h-4" />
            </button>
            <button onClick={onClose} className="btn-ghost p-2">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {loading && (
            <div className="flex items-center justify-center h-48 gap-3 text-sentinel-muted">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Loading memory...</span>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-xl bg-sentinel-red/5 border border-sentinel-red/20 text-sm text-sentinel-red">
              {error}
            </div>
          )}

          {/* L2 — Episodes */}
          {!loading && !error && layer === 'l2' && data && (
            <>
              {/* Dataset filter dropdown */}
              {datasets.length > 0 && (
                <div className="mb-3">
                  <select
                    value={filterDataset}
                    onChange={(e) => { setFilterDataset(e.target.value); load(e.target.value) }}
                    className="appearance-none bg-sentinel-card border border-sentinel-border rounded-lg
                               px-3 py-1.5 text-xs text-sentinel-text cursor-pointer w-full
                               hover:border-sentinel-blue/40 focus:border-sentinel-blue/60 focus:outline-none"
                  >
                    <option value="">All Datasets</option>
                    {datasets.map(ds => (
                      <option key={ds.filename} value={ds.filename}>{ds.filename}</option>
                    ))}
                  </select>
                </div>
              )}
              <L2Content episodes={data.episodes || []} />
            </>
          )}

          {/* L3 — Graph */}
          {!loading && !error && layer === 'l3' && data && (
            <L3Content graph={data} />
          )}

          {/* L4 — Patterns */}
          {!loading && !error && layer === 'l4' && data && (
            <L4Content patterns={data.patterns || []} />
          )}
        </div>
      </div>
    </div>
  )
}

// Dataset color palette (up to 8 datasets)
const DS_COLORS = [
  '#3B82F6', '#06B6D4', '#8B5CF6', '#10B981',
  '#F59E0B', '#EF4444', '#EC4899', '#14B8A6',
]

function datasetColor(dsName, allDatasets) {
  const idx = allDatasets.indexOf(dsName)
  return idx >= 0 ? DS_COLORS[idx % DS_COLORS.length] : '#6B7280'
}

// ── L2 Episodes ──────────────────────────────────────────────────────────────
function L2Content({ episodes }) {
  const [expanded, setExpanded] = useState(null)

  if (episodes.length === 0) {
    return <EmptyState message="No episodes yet. Run some queries to populate episodic memory." />
  }

  // Group by dataset
  const allDatasets = [...new Set(episodes.map(e => e.dataset || '').filter(Boolean))]
  const grouped = episodes.reduce((acc, ep, i) => {
    const key = ep.dataset || '__untagged__'
    if (!acc[key]) acc[key] = []
    acc[key].push({ ...ep, _origIdx: i })
    return acc
  }, {})
  const groups = Object.entries(grouped)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-sentinel-faint">{episodes.length} episodes stored</span>
        {allDatasets.length > 0 && (
          <div className="flex gap-2 flex-wrap">
            {allDatasets.map(ds => (
              <div key={ds} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: datasetColor(ds, allDatasets) }} />
                <span className="text-[10px] text-sentinel-faint truncate max-w-[100px]">{ds}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {groups.map(([dsKey, eps]) => {
        const isUntagged = dsKey === '__untagged__'
        const color = isUntagged ? '#6B7280' : datasetColor(dsKey, allDatasets)
        return (
          <div key={dsKey}>
            <div className="flex items-center gap-2 mb-1.5">
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: color }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color }}>
                {isUntagged ? 'Untagged' : dsKey}
              </span>
              <span className="text-[10px] text-sentinel-faint">({eps.length})</span>
            </div>
            <div className="space-y-1.5 pl-4">
              {eps.map((ep) => {
                const i = ep._origIdx
                return (
                  <div key={i} className="card overflow-hidden" style={{ borderLeftColor: color, borderLeftWidth: 2 }}>
                    <button
                      className="w-full flex items-start gap-3 p-3 text-left hover:bg-sentinel-card transition-colors"
                      onClick={() => setExpanded(expanded === i ? null : i)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-sentinel-text truncate">{ep.question}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-sentinel-faint">{ep.timestamp?.slice(0, 16) || 'Unknown time'}</span>
                          <span className="text-xs">·</span>
                          <span className="text-xs text-sentinel-green">Score: {(ep.score || 1.0).toFixed(2)}</span>
                        </div>
                      </div>
                      <div className="text-sentinel-faint mt-1">{expanded === i ? '▲' : '▼'}</div>
                    </button>
                    {expanded === i && (
                      <div className="px-3 pb-3 space-y-2 border-t border-sentinel-border">
                        {ep.sql && (
                          <div>
                            <div className="text-xs text-sentinel-faint mb-1 mt-2">SQL</div>
                            <pre className="sql-code text-xs max-h-32 overflow-auto">{ep.sql}</pre>
                          </div>
                        )}
                        {ep.result_summary && (
                          <div>
                            <div className="text-xs text-sentinel-faint mb-1">Result Summary</div>
                            <p className="text-xs text-sentinel-muted leading-relaxed">{ep.result_summary}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// -- L3 Graph -- Interactive SVG visualization --------------------------
const TYPE_CFG = {
  table: { fill: '#1E3A5F', stroke: '#3B82F6', text: '#60A5FA' },
  column: { fill: '#1A1F35', stroke: '#374155', text: '#6B7280' },
  business_rule: { fill: '#2D1F4F', stroke: '#8B5CF6', text: '#A78BFA' },
  other: { fill: '#1F2937', stroke: '#374151', text: '#9CA3AF' },
}
const REL_CFG = {
  causes: '#EF4444', reduces: '#F59E0B', foreign_key: '#3B82F6',
  has_column: '#1E293B', determines: '#8B5CF6', influences: '#06B6D4',
}

// Build per-dataset fill/stroke from the global palette
function buildDatasetCfg(datasets) {
  return datasets.reduce((acc, ds, i) => {
    const base = DS_COLORS[i % DS_COLORS.length]
    acc[ds] = { fill: base + '22', stroke: base, text: base }
    return acc
  }, {})
}

function L3Content({ graph }) {
  const { nodes = [], edges = [] } = graph
  const [selected, setSelected] = useState(null)
  const [showGraph, setShowGraph] = useState(true)
  const [positions, setPositions] = useState({})

  // Build dataset list for coloring
  const allDs = [...new Set(nodes.map(n => n.dataset || '').filter(Boolean))]
  const dsCfg = buildDatasetCfg(allDs)

  const byType = nodes.reduce((acc, n) => {
    const t = n.type || 'other'; if (!acc[t]) acc[t] = []; acc[t].push(n); return acc
  }, {})

  useEffect(() => {
    if (!nodes.length) return
    const W = 660, H = 460, CX = W / 2, CY = H / 2
    const types = Object.keys(byType)
    const pos = {}
    types.forEach((type, ti) => {
      const group = byType[type]
      const ga = (ti / types.length) * Math.PI * 2
      const gr = type === 'business_rule' ? 80 : type === 'table' ? 130 : 230
      const cx = CX + gr * Math.cos(ga), cy = CY + gr * Math.sin(ga)
      group.forEach((n, j) => {
        const a = (j / group.length) * Math.PI * 2
        const r = Math.max(35, 18 * Math.log(group.length + 1))
        pos[n.id] = {
          x: Math.max(30, Math.min(W - 30, cx + r * Math.cos(a))),
          y: Math.max(20, Math.min(H - 20, cy + r * Math.sin(a))),
        }
      })
    })
    setPositions(pos)
  }, [nodes.length])

  const visNodes = nodes.filter(n =>
    n.type === 'table' || n.type === 'business_rule' ||
    (n.type === 'column' && edges.some(e => (e.source === n.id || e.target === n.id) && (e.confidence || 0) > 0.5))
  ).slice(0, 50)
  const visIds = new Set(visNodes.map(n => n.id))
  const visEdges = edges.filter(e => visIds.has(e.source) && visIds.has(e.target)).slice(0, 80)
  const selNode = selected ? nodes.find(n => n.id === selected) : null
  const conEdges = selected ? edges.filter(e => e.source === selected || e.target === selected) : []

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-2">
        {[['Nodes', nodes.length, 'text-sentinel-purple'], ['Edges', edges.length, 'text-sentinel-blue'], ['Types', Object.keys(byType).length, 'text-sentinel-cyan']].map(([l, v, c]) => (
          <div key={l} className="p-3 rounded-xl bg-sentinel-card border border-sentinel-border text-center">
            <div className={`text-xl font-bold ${c}`}>{v}</div>
            <div className="text-xs text-sentinel-faint">{l}</div>
          </div>
        ))}
      </div>
      <div className="flex rounded-lg border border-sentinel-border overflow-hidden">
        {[['Graph View', true], ['List View', false]].map(([label, val]) => (
          <button key={String(val)} onClick={() => setShowGraph(val)}
            className={clsx('flex-1 py-2 text-xs font-medium transition-colors',
              showGraph === val ? 'bg-sentinel-purple/20 text-sentinel-purple' : 'text-sentinel-faint hover:text-sentinel-muted')}>
            {label}
          </button>
        ))}
      </div>
      {showGraph ? (
        <div className="rounded-xl border border-sentinel-border overflow-hidden bg-[#0A0E1A]">
          <div className="flex justify-between px-3 py-2 border-b border-sentinel-border">
            <span className="text-xs text-sentinel-faint">{visNodes.length} nodes &middot; {visEdges.length} edges</span>
            <span className="text-xs text-sentinel-faint italic">Click node for details</span>
          </div>
          <svg width="100%" viewBox="0 0 660 460" style={{ height: 460 }}>
            <defs>
              {Object.entries(REL_CFG).map(([rel, color]) => (
                <marker key={rel} id={`arr-${rel}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                  <path d="M0,0 L0,6 L8,3 z" fill={color} opacity="0.7" />
                </marker>
              ))}
            </defs>
            {visEdges.map((e, i) => {
              const sp = positions[e.source], tp = positions[e.target]
              if (!sp || !tp) return null
              const color = REL_CFG[e.rel] || REL_CFG.foreign_key
              const hi = selected && (e.source === selected || e.target === selected)
              return <line key={i} x1={sp.x} y1={sp.y} x2={tp.x} y2={tp.y}
                stroke={color} strokeWidth={hi ? 2 : 1} opacity={selected ? (hi ? 0.9 : 0.1) : 0.4}
                markerEnd={`url(#arr-${e.rel || 'foreign_key'})`} />
            })}
            {visNodes.map(node => {
              const pos = positions[node.id]; if (!pos) return null
              // Color by dataset if available, else fall back to type-based
              const dsCfgNode = node.dataset && dsCfg[node.dataset]
              const cfg = dsCfgNode || TYPE_CFG[node.type] || TYPE_CFG.other
              const isSel = selected === node.id
              const isCon = selected && conEdges.some(e => e.source === node.id || e.target === node.id)
              const op = selected ? (isSel || isCon ? 1 : 0.2) : 1
              const r = node.type === 'table' ? 16 : node.type === 'business_rule' ? 12 : 7
              const lbl = node.id.includes('.') ? node.id.split('.').pop() : node.id.replace(/_/g, ' ')
              return (
                <g key={node.id} transform={`translate(${pos.x},${pos.y})`} opacity={op}
                  style={{ cursor: 'pointer' }} onClick={() => setSelected(selected === node.id ? null : node.id)}>
                  <circle r={r} fill={cfg.fill} stroke={isSel ? '#F1F5F9' : cfg.stroke} strokeWidth={isSel ? 2.5 : 1.5} />
                  {node.type !== 'column' && (
                    <text textAnchor="middle" dy={r + 11} fontSize={node.type === 'table' ? 10 : 9}
                      fill={cfg.text} style={{ userSelect: 'none', pointerEvents: 'none' }}>
                      {lbl.length > 13 ? lbl.slice(0, 11) + '...' : lbl}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
          <div className="flex flex-wrap gap-3 px-3 py-2 border-t border-sentinel-border">
            {allDs.length > 0 ? (
              allDs.map(ds => (
                <div key={ds} className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full border" style={{ background: dsCfg[ds].fill, borderColor: dsCfg[ds].stroke }} />
                  <span className="text-xs text-sentinel-faint truncate max-w-[100px]">{ds}</span>
                </div>
              ))
            ) : (
              Object.entries(TYPE_CFG).filter(([t]) => t !== 'other').map(([type, cfg]) => (
                <div key={type} className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full border" style={{ background: cfg.fill, borderColor: cfg.stroke }} />
                  <span className="text-xs text-sentinel-faint">{type.replace(/_/g, ' ')}</span>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {Object.entries(byType).map(([type, items]) => (
            <div key={type}>
              <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">
                {type.replace(/_/g, ' ')} ({items.length})
              </div>
              <div className="space-y-1.5">
                {items.slice(0, 8).map(node => (
                  <div key={node.id} className="flex gap-2 p-2.5 rounded-lg bg-sentinel-card border border-sentinel-border">
                    <div className="w-1.5 h-1.5 rounded-full bg-sentinel-purple flex-shrink-0 mt-1.5" />
                    <div className="min-w-0">
                      <div className="text-xs font-medium text-sentinel-text truncate">{node.label}</div>
                      {node.description && <div className="text-xs text-sentinel-faint mt-0.5 line-clamp-2">{node.description}</div>}
                    </div>
                  </div>
                ))}
                {items.length > 8 && <div className="text-xs text-sentinel-faint pl-4">+{items.length - 8} more</div>}
              </div>
            </div>
          ))}
        </div>
      )}
      {selNode && (
        <div className="p-3 rounded-xl bg-sentinel-purple/5 border border-sentinel-purple/20">
          <div className="text-xs font-semibold text-sentinel-purple mb-1">{selNode.id}</div>
          <div className="text-xs text-sentinel-muted mb-2">Type: <span className="text-sentinel-text capitalize">{selNode.type}</span></div>
          {selNode.description && <div className="text-xs text-sentinel-muted mb-2">{selNode.description}</div>}
          {conEdges.length > 0 && (
            <div>
              <div className="text-xs text-sentinel-faint mb-1.5">Connections ({conEdges.length})</div>
              <div className="space-y-1">
                {conEdges.slice(0, 6).map((e, i) => {
                  const isSrc = e.source === selNode.id
                  const other = (isSrc ? e.target : e.source).split('.').pop()
                  const color = REL_CFG[e.rel] || '#6B7280'
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      {!isSrc && <span className="text-sentinel-faint">{other}</span>}
                      <span style={{ color }} className="font-medium">-&gt; {e.rel}</span>
                      {isSrc && <span className="text-sentinel-faint">{other}</span>}
                      {e.weight != null && (
                        <span className={`ml-auto font-mono ${Number(e.weight) > 0 ? 'text-sentinel-green' : 'text-sentinel-red'}`}>
                          {Number(e.weight).toFixed(2)}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
      {edges.filter(e => (e.confidence || 0) > 0.5).length > 0 && !selNode && (
        <div>
          <div className="text-xs font-semibold text-sentinel-faint uppercase tracking-wider mb-2">Causal Edges (sample)</div>
          <div className="space-y-1.5">
            {edges.filter(e => (e.confidence || 0) > 0.5).slice(0, 8).map((e, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border text-xs">
                <span className="text-sentinel-muted truncate max-w-[90px]">{e.source.split('.').pop()}</span>
                <span style={{ color: REL_CFG[e.rel] || '#6B7280' }} className="font-medium flex-shrink-0">-&gt; {e.rel}</span>
                <span className="text-sentinel-muted truncate max-w-[90px]">{e.target.split('.').pop()}</span>
                {e.weight != null && (
                  <span className={`ml-auto flex-shrink-0 font-mono ${Number(e.weight) > 0 ? 'text-sentinel-green' : 'text-sentinel-red'}`}>
                    {Number(e.weight).toFixed(2)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── L4 Patterns ───────────────────────────────────────────────────────────────
function L4Content({ patterns }) {
  const [expanded, setExpanded] = useState(null)

  if (patterns.length === 0) {
    return <EmptyState message="No SQL patterns yet. Patterns are promoted from high-scoring L2 episodes." />
  }

  return (
    <div className="space-y-2">
      <div className="text-xs text-sentinel-faint mb-3">{patterns.length} patterns stored</div>
      {patterns.map((pat, i) => (
        <div key={i} className="card overflow-hidden">
          <button
            className="w-full flex items-center gap-3 p-3 text-left hover:bg-sentinel-card transition-colors"
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <Code2 className="w-4 h-4 text-sentinel-cyan flex-shrink-0" />
            <span className="flex-1 text-sm font-medium text-sentinel-text truncate">{pat.problem_type}</span>
            <span className="text-sentinel-faint">{expanded === i ? '▲' : '▼'}</span>
          </button>
          {expanded === i && (
            <div className="px-3 pb-3 border-t border-sentinel-border">
              {pat.example_query && (
                <div className="mt-2 mb-2 text-xs text-sentinel-faint">{pat.example_query}</div>
              )}
              {pat.sql_template && (
                <pre className="sql-code text-xs max-h-48 overflow-auto">{pat.sql_template}</pre>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
      <div className="w-12 h-12 rounded-xl bg-sentinel-card border border-sentinel-border flex items-center justify-center">
        <Database className="w-6 h-6 text-sentinel-faint" />
      </div>
      <p className="text-sm text-sentinel-muted max-w-xs">{message}</p>
    </div>
  )
}
