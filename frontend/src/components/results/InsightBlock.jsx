import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertCircle } from 'lucide-react'

// Custom markdown renderers styled to match the dark SENTINEL theme
const MD_COMPONENTS = {
  // Headings
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-sentinel-text mb-2 mt-3 first:mt-0">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-bold text-sentinel-text mb-1.5 mt-3 first:mt-0 pb-1 border-b border-sentinel-border">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-sentinel-text mb-1 mt-2 first:mt-0">{children}</h3>
  ),
  // Paragraphs
  p: ({ children }) => (
    <p className="text-sm text-sentinel-muted leading-relaxed mb-2 last:mb-0">{children}</p>
  ),
  // Bold — highlight numbers and key terms
  strong: ({ children }) => (
    <strong className="font-semibold text-sentinel-text">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-sentinel-muted">{children}</em>
  ),
  // Lists
  ul: ({ children }) => (
    <ul className="space-y-1 mb-2 pl-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="space-y-1 mb-2 pl-1 list-decimal list-inside">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="text-sm text-sentinel-muted leading-relaxed flex gap-2">
      <span className="text-sentinel-blue flex-shrink-0 mt-0.5">•</span>
      <span>{children}</span>
    </li>
  ),
  // Code — Python blocks are executed on the backend and shown as interactive
  // Plotly charts; replace them with a pill instead of raw code.
  code: ({ inline, className, children }) => {
    const lang = (className || '').replace('language-', '').toLowerCase()
    if (!inline && (lang === 'python' || lang === 'py')) {
      return (
        <div className="flex items-center gap-2 my-2 px-3 py-2 rounded-lg bg-sentinel-card border border-sentinel-border text-xs text-sentinel-blue">
          <span>⚡</span>
          <span className="italic text-sentinel-muted">Chart rendered in Visualisations above</span>
        </div>
      )
    }
    return inline ? (
      <code className="px-1.5 py-0.5 rounded text-xs font-mono bg-sentinel-card border border-sentinel-border text-sentinel-blue">
        {children}
      </code>
    ) : (
      <pre className="text-xs font-mono bg-sentinel-card border border-sentinel-border rounded-lg p-3 overflow-auto my-2 text-sentinel-muted whitespace-pre-wrap">
        {children}
      </pre>
    )
  },
  // Block quote — used for alerts and summaries
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-sentinel-blue pl-3 my-2 text-sm text-sentinel-muted italic">
      {children}
    </blockquote>
  ),
  // Horizontal rule — section separator
  hr: () => <hr className="border-sentinel-border my-3" />,
  // Tables (GFM)
  table: ({ children }) => (
    <div className="overflow-x-auto my-2 rounded-lg border border-sentinel-border">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-sentinel-card text-sentinel-faint uppercase tracking-wider">{children}</thead>
  ),
  tbody: ({ children }) => <tbody className="divide-y divide-sentinel-border">{children}</tbody>,
  tr: ({ children }) => <tr className="hover:bg-sentinel-card/50 transition-colors">{children}</tr>,
  th: ({ children }) => <th className="px-3 py-2 text-left font-semibold">{children}</th>,
  td: ({ children }) => <td className="px-3 py-2 text-sentinel-muted">{children}</td>,
}

export default function InsightBlock({ text, error }) {
  if (error) {
    return (
      <div className="flex items-start gap-3 p-4 rounded-xl bg-sentinel-red/5 border border-sentinel-red/20">
        <AlertCircle className="w-5 h-5 text-sentinel-red flex-shrink-0 mt-0.5" />
        <div>
          <div className="text-sm font-medium text-sentinel-red mb-1">Analysis Error</div>
          <div className="text-sm text-sentinel-muted">{error}</div>
        </div>
      </div>
    )
  }

  if (!text) return null

  return (
    <div className="space-y-1">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={MD_COMPONENTS}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
