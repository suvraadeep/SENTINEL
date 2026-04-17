import React from 'react'
import clsx from 'clsx'

// Provider logos — map to public/logos/ paths
const LOGO_MAP = {
  openai:       '/logos/openai.png',
  google:       '/logos/gemini.webp',
  anthropic:    '/logos/claude.svg',
  groq:         '/logos/groq-icon.webp',
  nvidia:       '/logos/Nvidia_logo.svg.png',
  huggingface:  '/logos/huggingface.svg',
}

const PROVIDER_DESCRIPTIONS = {
  openai:       'GPT-4o, o3 and more',
  google:       'Gemini 2.0 & 1.5 series',
  anthropic:    'Claude Opus, Sonnet, Haiku',
  groq:         'Ultra-fast inference',
  nvidia:       'NIM AI endpoints',
  huggingface:  'Open-source models via HF Hub',
}

const PROVIDER_ACCENT = {
  openai:       'hover:border-green-500/50   data-[selected]:border-green-500',
  google:       'hover:border-blue-400/50    data-[selected]:border-blue-400',
  anthropic:    'hover:border-orange-500/50  data-[selected]:border-orange-500',
  groq:         'hover:border-sentinel-cyan/50 data-[selected]:border-sentinel-cyan',
  nvidia:       'hover:border-sentinel-green/50 data-[selected]:border-sentinel-green',
  huggingface:  'hover:border-yellow-500/50  data-[selected]:border-yellow-500',
}

export default function OperatorGrid({ selected, onSelect, catalogue }) {
  const providers = Object.keys(catalogue).length > 0
    ? Object.keys(catalogue)
    : ['openai', 'google', 'anthropic', 'groq', 'nvidia', 'huggingface']

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
      {providers.map((prov) => {
        const isSelected = selected === prov
        const cat = catalogue[prov] || {}
        return (
          <button
            key={prov}
            onClick={() => onSelect(prov)}
            data-selected={isSelected || undefined}
            className={clsx(
              'relative flex flex-col items-center gap-3 p-5 rounded-2xl',
              'bg-sentinel-surface border-2 cursor-pointer',
              'transition-all duration-200 group',
              PROVIDER_ACCENT[prov] || 'hover:border-sentinel-blue/50',
              isSelected
                ? 'border-sentinel-blue bg-sentinel-card shadow-glow-blue'
                : 'border-sentinel-border'
            )}
          >
            {/* Selected indicator */}
            {isSelected && (
              <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-sentinel-blue flex items-center justify-center">
                <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            )}

            {/* Logo */}
            <div className="w-12 h-12 flex items-center justify-center">
              <img
                src={LOGO_MAP[prov]}
                alt={cat.display_name || prov}
                className="max-w-full max-h-full object-contain"
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.nextSibling.style.display = 'flex'
                }}
              />
              <div className="hidden w-10 h-10 rounded-lg bg-sentinel-card border border-sentinel-border items-center justify-center">
                <span className="text-lg font-bold text-sentinel-muted">
                  {(cat.display_name || prov)[0].toUpperCase()}
                </span>
              </div>
            </div>

            {/* Name */}
            <div className="text-center">
              <div className={clsx(
                'text-sm font-semibold transition-colors',
                isSelected ? 'text-sentinel-text' : 'text-sentinel-muted group-hover:text-sentinel-text'
              )}>
                {cat.display_name || prov}
              </div>
              <div className="text-xs text-sentinel-faint mt-0.5">
                {PROVIDER_DESCRIPTIONS[prov] || ''}
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
