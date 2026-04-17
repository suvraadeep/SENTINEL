import React from 'react'
import { ChevronDown } from 'lucide-react'

export default function ModelSelector({ provider, catalogue, mainModel, fastModel, onMainChange, onFastChange }) {
  const cat = catalogue[provider] || {}
  const models = cat.models || []

  if (!provider || models.length === 0) {
    return (
      <div className="p-4 rounded-xl bg-sentinel-card border border-sentinel-border text-sm text-sentinel-muted text-center">
        Select a provider first to see available models.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Primary model */}
      <div>
        <label className="block text-xs font-medium text-sentinel-muted mb-2 uppercase tracking-wider">
          Primary Model (Analysis & SQL)
        </label>
        <div className="relative">
          <select
            value={mainModel}
            onChange={(e) => onMainChange(e.target.value)}
            className="input appearance-none pr-10 cursor-pointer"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sentinel-faint pointer-events-none" />
        </div>
      </div>

      {/* Fast model */}
      <div>
        <label className="block text-xs font-medium text-sentinel-muted mb-2 uppercase tracking-wider">
          Fast Model (Summaries & Classification)
        </label>
        <div className="relative">
          <select
            value={fastModel}
            onChange={(e) => onFastChange(e.target.value)}
            className="input appearance-none pr-10 cursor-pointer"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sentinel-faint pointer-events-none" />
        </div>
        <p className="text-xs text-sentinel-faint mt-1.5">
          Used for quick classifications and summaries to reduce latency.
        </p>
      </div>
    </div>
  )
}
