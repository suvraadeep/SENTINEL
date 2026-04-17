import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../../context/AppContext'
import { switchVersion } from '../../api/client'
import { Database, LogOut, Trash2, RefreshCw, Layers } from 'lucide-react'
import clsx from 'clsx'

export default function TopBar() {
  const navigate = useNavigate()
  const {
    dataSource, isQuerying,
    provider, mainModel,
    clearChat, refreshMemory,
    setIsReady,
    activeDataset,
    activeVersion, setActiveVersion,
    hasModified,
  } = useApp()

  const [switching, setSwitching] = useState(false)

  const handleVersionToggle = async (version) => {
    if (version === activeVersion || switching) return
    setSwitching(true)
    try {
      const res = await switchVersion(version)
      if (res.success) {
        setActiveVersion(version)
      }
    } catch (err) {
      console.error('Switch version failed:', err)
    } finally {
      setSwitching(false)
    }
  }

  const handleDisconnect = () => {
    setIsReady(false)
    navigate('/login')
  }

  return (
    <header className="flex items-center justify-between h-14 px-5 border-b border-sentinel-border bg-sentinel-surface/80 backdrop-blur-sm flex-shrink-0">

      {/* Data source + Version toggle */}
      <div className="flex items-center gap-3">
        <div className={`glow-dot ${isQuerying ? 'bg-sentinel-yellow' : 'bg-sentinel-green'}`} />
        <Database className="w-4 h-4 text-sentinel-faint" />

        {/* Dataset name — clean display only */}
        <span className="text-sm text-sentinel-muted font-medium truncate" style={{ maxWidth: 240 }}>
          {activeDataset?.replace(/\.[^.]+$/, '') || dataSource}
        </span>

        {/* Version Toggle (Original / Modified) — styled pill toggle */}
        <div className="w-px h-6 bg-sentinel-border" />
        <div className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-sentinel-cyan" />
          <div className={clsx(
            'flex rounded-lg overflow-hidden border-2 transition-all duration-300',
            switching ? 'opacity-50 pointer-events-none' : '',
            activeVersion === 'modified' && hasModified
              ? 'border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
              : 'border-indigo-500/40 shadow-[0_0_8px_rgba(99,102,241,0.1)]',
          )}>
            {['original', 'modified'].map(v => (
              <button
                key={v}
                onClick={() => handleVersionToggle(v)}
                disabled={!hasModified || switching}
                className={clsx(
                  'px-3.5 py-1.5 text-xs font-bold transition-all duration-200 capitalize',
                  activeVersion === v
                    ? v === 'modified'
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : 'bg-indigo-500/20 text-indigo-300'
                    : !hasModified
                      ? 'bg-sentinel-card text-sentinel-faint/40 cursor-not-allowed'
                      : 'bg-sentinel-card text-sentinel-faint hover:text-sentinel-muted hover:bg-sentinel-hover cursor-pointer',
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

        {isQuerying && (
          <span className="badge-yellow ml-1">
            <span className="animate-pulse">Processing...</span>
          </span>
        )}
      </div>

      {/* Session info + actions */}
      <div className="flex items-center gap-2">
        {mainModel && (
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-sentinel-card border border-sentinel-border">
            <div className="w-1.5 h-1.5 rounded-full bg-sentinel-blue" />
            <span className="text-xs text-sentinel-muted capitalize">{provider}</span>
            <span className="text-xs text-sentinel-faint">·</span>
            <span className="text-xs text-sentinel-faint">{mainModel.split('/').pop()}</span>
          </div>
        )}

        <button
          onClick={refreshMemory}
          className="btn-ghost"
          title="Refresh memory stats"
        >
          <RefreshCw className="w-4 h-4" />
        </button>

        <button
          onClick={clearChat}
          className="btn-ghost"
          title="Clear chat"
        >
          <Trash2 className="w-4 h-4" />
        </button>

        <button
          onClick={handleDisconnect}
          className="btn-ghost text-sentinel-red hover:text-sentinel-red hover:bg-sentinel-red/10"
          title="Disconnect provider"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}
