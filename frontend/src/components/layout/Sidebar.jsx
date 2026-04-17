import React, { useState } from 'react'
import clsx from 'clsx'
import { useApp } from '../../context/AppContext'
import MemoryBar from '../memory/MemoryBar'
import {
  Brain, BarChart3, Database,
  ChevronRight, Zap, FlaskConical, Table2, ChevronDown,
  Trash2, Loader, AlertTriangle, GitBranch,
} from 'lucide-react'

const NAV_ITEMS = [
  { id: 'intelligence', label: 'Intelligence',  icon: Brain,          desc: 'Query & Analysis' },
  { id: 'analytics',   label: 'Insights Hub',  icon: BarChart3,      desc: 'History · Signals · Charts' },
  { id: 'anomaly',     label: 'Anomaly',        icon: AlertTriangle,  desc: 'Anomaly Detection' },
  { id: 'rca',         label: 'RCA',            icon: GitBranch,      desc: 'Causal Root Cause' },
  { id: 'datalab',     label: 'DataLab',        icon: FlaskConical,   desc: 'Explore & Transform' },
]

function DatasetItem({ dataset, onOpen, onRemove }) {
  const [removing, setRemoving] = useState(false)

  const handleRemove = async (e) => {
    e.stopPropagation()
    if (!window.confirm(`Remove "${dataset.filename}" and drop all its tables?`)) return
    setRemoving(true)
    await onRemove(dataset.filename)
    setRemoving(false)
  }

  return (
    <div className="group flex items-center gap-1 rounded-lg hover:bg-sentinel-card transition-colors">
      <button
        onClick={onOpen}
        className="flex-1 flex items-center gap-2 px-2 py-1.5 text-left min-w-0"
      >
        <Table2 className="w-3 h-3 text-sentinel-faint flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium text-sentinel-muted truncate group-hover:text-sentinel-text transition-colors">
            {dataset.filename}
          </div>
          <div className="text-[10px] text-sentinel-faint">
            {dataset.row_count?.toLocaleString() ?? '?'} rows · {dataset.tables?.length ?? 1} table{dataset.tables?.length !== 1 ? 's' : ''}
          </div>
        </div>
      </button>
      <button
        onClick={handleRemove}
        disabled={removing}
        title={`Remove ${dataset.filename}`}
        className="opacity-0 group-hover:opacity-100 p-1.5 mr-1 rounded text-sentinel-faint hover:text-sentinel-red hover:bg-sentinel-red/10 transition-all flex-shrink-0"
      >
        {removing ? <Loader className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
      </button>
    </div>
  )
}

export default function Sidebar() {
  const {
    activePage, setActivePage,
    memoryStats, setMemoryModalLayer,
    provider, mainModel,
    datasets, removeDataset,
    openDataLab,
  } = useApp()

  const [datasetsExpanded, setDatasetsExpanded] = useState(true)

  return (
    <aside className="flex flex-col w-60 bg-sentinel-surface border-r border-sentinel-border flex-shrink-0 overflow-hidden">

      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-sentinel-border">
        <div className="flex-shrink-0">
          <img
            src="/logos/sentinels_logo.png?v=2"
            alt="SENTINEL"
            className="w-8 h-8 object-contain"
            onError={(e) => { e.target.style.display='none' }}
          />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-bold text-gradient-blue">SENTINEL</div>
          <div className="text-xs text-sentinel-faint truncate">Analytics Intelligence</div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 space-y-0.5 px-2 overflow-y-auto no-scrollbar">
        <div className="px-2 py-1 mb-2">
          <span className="section-label">Navigation</span>
        </div>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          const active = activePage === item.id
          const isDataLab = item.id === 'datalab'
          return (
            <button
              key={item.id}
              onClick={() => isDataLab ? openDataLab() : setActivePage(item.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-150 group',
                active
                  ? 'bg-sentinel-blue/10 border border-sentinel-blue/20 text-sentinel-text'
                  : 'text-sentinel-muted hover:bg-sentinel-card hover:text-sentinel-text'
              )}
            >
              <Icon className={clsx(
                'w-4 h-4 flex-shrink-0',
                active
                  ? isDataLab ? 'text-sentinel-cyan' : 'text-sentinel-blue'
                  : 'text-sentinel-faint group-hover:text-sentinel-muted'
              )} />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">{item.label}</div>
              </div>
              {active && !isDataLab && <ChevronRight className="w-3 h-3 text-sentinel-blue" />}
              {isDataLab && (
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-sentinel-cyan/10 text-sentinel-cyan font-semibold">
                  {datasets.length > 0 ? datasets.length : 'NEW'}
                </span>
              )}
            </button>
          )
        })}

        {/* Datasets section */}
        {datasets.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setDatasetsExpanded(v => !v)}
              className="w-full flex items-center justify-between px-2 py-1 mb-1"
            >
              <span className="section-label">Datasets ({datasets.length})</span>
              <ChevronDown className={clsx('w-3 h-3 text-sentinel-faint transition-transform', datasetsExpanded ? 'rotate-180' : '')} />
            </button>
            {datasetsExpanded && (
              <div className="space-y-0.5 px-1">
                {datasets.map((ds) => (
                  <DatasetItem
                    key={ds.filename}
                    dataset={ds}
                    onOpen={() => openDataLab(ds.tables?.[0] ?? null)}
                    onRemove={removeDataset}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </nav>

      {/* Memory Section */}
      <div className="border-t border-sentinel-border p-3">
        <div className="flex items-center justify-between mb-3 px-1">
          <span className="section-label">Memory Units</span>
          <Zap className="w-3 h-3 text-sentinel-yellow" />
        </div>
        <div className="space-y-2">
          <MemoryBar
            label="L2 Episodic"
            count={memoryStats?.l2?.count ?? 0}
            pct={memoryStats?.l2?.pct ?? 0}
            color="bg-sentinel-blue"
            onClick={() => setMemoryModalLayer('l2')}
            tooltip="Past query cache (semantic similarity)"
          />
          <MemoryBar
            label="L3 Causal Graph"
            count={`${memoryStats?.l3?.nodes ?? 0}n`}
            pct={Math.min(100, ((memoryStats?.l3?.nodes ?? 0) / 200) * 100)}
            color="bg-sentinel-purple"
            onClick={() => setMemoryModalLayer('l3')}
            tooltip="Causal knowledge graph"
          />
          <MemoryBar
            label="L4 Procedural"
            count={memoryStats?.l4?.count ?? 0}
            pct={memoryStats?.l4?.pct ?? 0}
            color="bg-sentinel-cyan"
            onClick={() => setMemoryModalLayer('l4')}
            tooltip="SQL template patterns"
          />
        </div>
      </div>

      {/* Provider tag */}
      {provider && (
        <div className="px-3 pb-3">
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border">
            <Database className="w-3.5 h-3.5 text-sentinel-faint" />
            <div className="flex-1 min-w-0">
              <div className="text-xs text-sentinel-faint">Provider</div>
              <div className="text-xs font-medium text-sentinel-muted truncate capitalize">{provider} · {mainModel?.split('/').pop()}</div>
            </div>
            <div className="glow-dot bg-sentinel-green" />
          </div>
        </div>
      )}
    </aside>
  )
}
