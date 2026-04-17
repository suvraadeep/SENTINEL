import React from 'react'
import clsx from 'clsx'

export default function MemoryBar({ label, count, pct, color, onClick, tooltip }) {
  const pctNum = Math.min(100, Math.max(0, typeof pct === 'number' ? pct : 0))

  return (
    <button
      onClick={onClick}
      title={tooltip}
      className="memory-layer w-full text-left"
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-sentinel-muted font-medium truncate">{label}</span>
          <span className="text-xs text-sentinel-faint ml-2 flex-shrink-0">{count}</span>
        </div>
        <div className="progress-bar">
          <div
            className={clsx('progress-fill', color)}
            style={{ width: `${pctNum}%` }}
          />
        </div>
        <div className="text-xs text-sentinel-faint mt-1">{pctNum.toFixed(0)}% capacity</div>
      </div>
    </button>
  )
}
