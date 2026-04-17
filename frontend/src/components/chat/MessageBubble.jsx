import React from 'react'
import { Brain, User } from 'lucide-react'
import ResultCard from './ResultCard'
import clsx from 'clsx'

export default function MessageBubble({ message }) {
  const { role, content, result, timestamp } = message
  const isUser = role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-in">
        <div className="flex items-end gap-2 max-w-2xl">
          <div className="bg-sentinel-blue text-white px-4 py-3 rounded-2xl rounded-br-md text-sm leading-relaxed">
            {content}
          </div>
          <div className="w-7 h-7 rounded-full bg-sentinel-card border border-sentinel-border flex items-center justify-center flex-shrink-0 mb-0.5">
            <User className="w-4 h-4 text-sentinel-muted" />
          </div>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex items-start gap-3 animate-slide-in">
      <div className="w-8 h-8 rounded-lg bg-sentinel-blue/10 border border-sentinel-blue/20 flex items-center justify-center flex-shrink-0 mt-1">
        <Brain className="w-4 h-4 text-sentinel-blue" />
      </div>
      <div className="flex-1 min-w-0">
        {result ? (
          <ResultCard result={result} />
        ) : (
          <div className="card p-4 text-sm text-sentinel-muted">
            {content || 'No response'}
          </div>
        )}
        {timestamp && (
          <div className="text-xs text-sentinel-faint mt-1 pl-1">
            {new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            {result?.duration_ms && <span className="ml-2">· {(result.duration_ms / 1000).toFixed(1)}s</span>}
          </div>
        )}
      </div>
    </div>
  )
}
