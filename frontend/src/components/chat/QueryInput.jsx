import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { useApp } from '../../context/AppContext'
import { runQuery, uploadFile } from '../../api/client'
import clsx from 'clsx'

export default function QueryInput() {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)
  const { addMessage, isQuerying, setIsQuerying, setDataSource, setUploadedFile, addDataset, refreshMemory, activeDataset, activeVersion, datasets, setActivePage } = useApp()

  // Intent patterns — redirect to dedicated pages rather than chat
  const ANOMALY_RE = /\b(anomaly|anomalies|outlier|outliers|detect\s+anomal|anomal\s+detect|find\s+anomal|show\s+anomal|run\s+anomaly|anomaly\s+scan)\b/i
  const RCA_RE     = /\b(rca|root.?cause|causal\s+analysis|run\s+rca|causal\s+graph|what\s+caused|why\s+(did|has|have))\b/i

  // Listen for example query fill events
  useEffect(() => {
    const handler = (e) => {
      setText(e.detail)
      textareaRef.current?.focus()
    }
    window.addEventListener('sentinel:fill-query', handler)
    return () => window.removeEventListener('sentinel:fill-query', handler)
  }, [])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [text])

  const handleSubmit = async () => {
    const query = text.trim()
    if (!query || isQuerying) return

    // Redirect to dedicated pages for anomaly/RCA intents
    if (ANOMALY_RE.test(query)) {
      setText('')
      setActivePage('anomaly')
      return
    }
    if (RCA_RE.test(query)) {
      setText('')
      setActivePage('rca')
      return
    }

    setText('')
    setIsQuerying(true)

    // Resolve dataset based on version toggle.
    // If 'modified' is selected, route to the modified.csv virtual dataset.
    let targetDataset = activeDataset || null
    if (activeVersion === 'modified' && activeDataset) {
      const modDs = datasets.find(ds => ds.filename === 'modified.csv')
      if (modDs) {
        targetDataset = modDs.filename
      }
    }

    // Add user message (tagged with dataset for later cleanup)
    addMessage({ role: 'user', content: query, timestamp: Date.now(), dataset: targetDataset })

    try {
      const result = await runQuery(query, targetDataset)
      addMessage({
        role: 'assistant',
        content: result.insights || result.error || 'Analysis complete.',
        result,
        query,
        timestamp: Date.now(),
        dataset:  targetDataset,
        version:  activeVersion || 'original',
      })
      // Refresh memory after query
      refreshMemory()
    } catch (err) {
      addMessage({
        role: 'assistant',
        content: err?.response?.data?.detail || 'An error occurred. Please try again.',
        timestamp: Date.now(),
        dataset: targetDataset,
      })
    } finally {
      setIsQuerying(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    fileInputRef.current.value = ''
    try {
      addMessage({ role: 'user', content: `Uploading dataset: ${file.name}`, timestamp: Date.now() })
      const result = await uploadFile(file)
      setDataSource(file.name.replace(/\.[^.]+$/, ''))
      setUploadedFile(file)

      // Register dataset in global context for sidebar + DataLab
      addDataset({
        filename: file.name,
        tables: result.tables ?? [],
        row_count: result.row_count ?? 0,
        date_min: result.date_min,
        date_max: result.date_max,
      })

      const isAdditional = result.dataset_count > 1
      const joinHint = isAdditional
        ? ` (${result.dataset_count} datasets loaded — you can JOIN across them!)`
        : ''
      addMessage({
        role: 'assistant',
        content: `Dataset loaded: **${file.name}** — ${result.row_count?.toLocaleString()} rows, ${result.tables?.length} table(s).${joinHint} You can now query it!`,
        timestamp: Date.now(),
      })
    } catch (err) {
      addMessage({
        role: 'assistant',
        content: `Upload failed: ${err?.response?.data?.detail || 'Unknown error'}`,
        timestamp: Date.now(),
      })
    }
  }

  return (
    <div className="px-6 py-4 border-t border-sentinel-border bg-sentinel-surface/80 backdrop-blur-sm">
      <div className={clsx(
        'flex items-end gap-3 p-3 rounded-2xl border transition-all duration-200',
        isQuerying
          ? 'bg-sentinel-card border-sentinel-blue/30'
          : 'bg-sentinel-card border-sentinel-border focus-within:border-sentinel-blue/50'
      )}>

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask SENTINEL anything about your data... (Enter to send, Shift+Enter for new line)"
          rows={1}
          disabled={isQuerying}
          className="flex-1 bg-transparent resize-none outline-none text-sm text-sentinel-text
                     placeholder-sentinel-faint leading-relaxed py-1 min-h-[32px] max-h-40"
        />

        {/* Send button */}
        <button
          onClick={handleSubmit}
          disabled={!text.trim() || isQuerying}
          className={clsx(
            'flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-150',
            text.trim() && !isQuerying
              ? 'bg-sentinel-blue text-white hover:bg-blue-500 active:scale-95'
              : 'bg-sentinel-hover text-sentinel-faint cursor-not-allowed'
          )}
        >
          {isQuerying
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <Send className="w-4 h-4" />
          }
        </button>
      </div>
      <p className="text-xs text-sentinel-faint text-center mt-2">
        SENTINEL can run SQL, detect anomalies, forecast trends, and explain root causes.
      </p>
    </div>
  )
}
