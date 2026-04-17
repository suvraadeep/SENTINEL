import React, { useEffect, useRef } from 'react'
import { useApp } from '../../context/AppContext'
import MessageBubble from './MessageBubble'
import { Brain, Sparkles } from 'lucide-react'

const EXAMPLE_QUERIES = [
  'Show total revenue by category with trend analysis',
  'Detect anomalies in order volume across cities',
  'Forecast revenue for next 7 days with confidence intervals',
  'What are the root causes of the Electronics category revenue drop?',
  'Analyze price elasticity and customer lifetime value by loyalty tier',
]

export default function ChatWindow() {
  const { messages, isQuerying } = useApp()
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isQuerying])

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">

      {/* Empty state */}
      {messages.length === 0 && !isQuerying && (
        <div className="flex flex-col items-center justify-center h-full gap-6 py-12">
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-sentinel-blue/10 border border-sentinel-blue/20 flex items-center justify-center">
              <Brain className="w-8 h-8 text-sentinel-blue" />
            </div>
            <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-sentinel-cyan flex items-center justify-center">
              <Sparkles className="w-3 h-3 text-white" />
            </div>
          </div>
          <div className="text-center max-w-md">
            <h3 className="text-lg font-semibold text-sentinel-text mb-2">
              Ask SENTINEL Anything
            </h3>
            <p className="text-sm text-sentinel-muted leading-relaxed">
              Query your data using natural language. SENTINEL will generate SQL,
              create visualizations, detect anomalies, and perform causal analysis automatically.
            </p>
          </div>

          {/* Example queries */}
          <div className="w-full max-w-lg space-y-2">
            <p className="text-xs text-sentinel-faint text-center uppercase tracking-wider mb-3">Try asking</p>
            {EXAMPLE_QUERIES.map((q, i) => (
              <ExampleQuery key={i} query={q} />
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {/* Loading indicator */}
      {isQuerying && (
        <div className="flex items-start gap-3 animate-fade-in">
          <div className="w-8 h-8 rounded-lg bg-sentinel-blue/10 border border-sentinel-blue/20 flex items-center justify-center flex-shrink-0">
            <Brain className="w-4 h-4 text-sentinel-blue" />
          </div>
          <div className="card p-4 max-w-sm">
            <div className="flex items-center gap-2 text-sm text-sentinel-muted mb-2">
              <span>SENTINEL is analysing</span>
              <div className="flex gap-1">
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-sentinel-blue inline-block" />
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-sentinel-blue inline-block" />
                <span className="typing-dot w-1.5 h-1.5 rounded-full bg-sentinel-blue inline-block" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="h-2 skeleton rounded w-3/4" />
              <div className="h-2 skeleton rounded w-1/2" />
              <div className="h-2 skeleton rounded w-5/6" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}

function ExampleQuery({ query }) {
  const { addMessage, setIsQuerying } = useApp()
  // The example queries don't actually trigger — they just show text
  // to guide the user. Clicking sets the input box via a custom event.
  const handleClick = () => {
    window.dispatchEvent(new CustomEvent('sentinel:fill-query', { detail: query }))
  }

  return (
    <button
      onClick={handleClick}
      className="w-full text-left px-4 py-2.5 rounded-xl bg-sentinel-card border border-sentinel-border
                 text-sm text-sentinel-muted hover:text-sentinel-text hover:border-sentinel-blue/30
                 hover:bg-sentinel-hover transition-all duration-150 truncate"
    >
      {query}
    </button>
  )
}
