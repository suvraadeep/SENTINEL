import React, { useRef, useEffect, useState } from 'react'
import { Maximize2, Download } from 'lucide-react'

export default function ChartFrame({ chart }) {
  const { title, html } = chart
  const iframeRef = useRef(null)
  const [expanded, setExpanded] = useState(false)
  const [height, setHeight] = useState(420)

  // Inject dark background into chart HTML
  const styledHtml = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    background: #111827 !important;
    overflow: hidden;
    font-family: Inter, sans-serif;
  }
  .plotly-graph-div { background: #111827 !important; }
</style>
</head>
<body>
${html}
</body>
</html>`

  const handleExpand = () => setExpanded((v) => !v)

  return (
    <div className="rounded-xl border border-sentinel-border overflow-hidden bg-sentinel-surface">
      {/* Chart header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-sentinel-border">
        <span className="text-sm font-medium text-sentinel-muted">{title || 'Chart'}</span>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExpand}
            className="btn-ghost p-1.5"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Plotly chart in sandboxed iframe */}
      <iframe
        ref={iframeRef}
        srcDoc={styledHtml}
        title={title || 'Chart'}
        className="w-full border-0"
        style={{ height: expanded ? '640px' : `${height}px`, background: '#111827' }}
        sandbox="allow-scripts allow-same-origin"
        scrolling="no"
      />
    </div>
  )
}
