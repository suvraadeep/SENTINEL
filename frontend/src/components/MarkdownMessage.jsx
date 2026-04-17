import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

/**
 * MarkdownMessage — shared dark-theme Markdown renderer for all chat panels.
 * Supports GFM (tables, strikethrough, task lists), LaTeX inline/block via KaTeX.
 */
export default function MarkdownMessage({ content, className = '' }) {
  return (
    <div className={`markdown-message text-sm leading-relaxed ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-base font-bold text-sentinel-text mt-3 mb-1 border-b border-sentinel-border pb-1">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-sm font-bold text-sentinel-text mt-3 mb-1">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-sentinel-blue mt-2 mb-1">
              {children}
            </h3>
          ),

          // Paragraphs
          p: ({ children }) => (
            <p className="text-sentinel-text mb-2 last:mb-0">{children}</p>
          ),

          // Strong / Em
          strong: ({ children }) => (
            <strong className="font-semibold text-white">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-sentinel-muted">{children}</em>
          ),

          // Code — inline
          code: ({ inline, className: cls, children, ...props }) => {
            if (inline) {
              return (
                <code
                  className="px-1 py-0.5 rounded text-xs font-mono bg-sentinel-hover text-sentinel-blue border border-sentinel-border"
                  {...props}
                >
                  {children}
                </code>
              )
            }
            return (
              <code
                className="block text-xs font-mono text-sentinel-text whitespace-pre-wrap"
                {...props}
              >
                {children}
              </code>
            )
          },

          // Code block wrapper
          pre: ({ children }) => (
            <pre className="bg-sentinel-hover border border-sentinel-border rounded-lg p-3 my-2 overflow-x-auto text-xs font-mono">
              {children}
            </pre>
          ),

          // Blockquote
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-sentinel-blue pl-3 my-2 text-sentinel-muted italic">
              {children}
            </blockquote>
          ),

          // Lists
          ul: ({ children }) => (
            <ul className="list-disc list-inside space-y-0.5 my-1.5 text-sentinel-text pl-2">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside space-y-0.5 my-1.5 text-sentinel-text pl-2">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-sentinel-text leading-relaxed">{children}</li>
          ),

          // Tables
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-lg border border-sentinel-border">
              <table className="w-full text-xs">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-sentinel-hover">{children}</thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-sentinel-border">{children}</tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-sentinel-hover/50 transition-colors">{children}</tr>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left font-semibold text-sentinel-text border-b border-sentinel-border">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-sentinel-muted">{children}</td>
          ),

          // Horizontal rule
          hr: () => <hr className="border-sentinel-border my-3" />,

          // Links
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sentinel-blue hover:underline"
            >
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
