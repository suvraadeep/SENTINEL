import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { getMemoryStats, getDatasets, removeDataset as apiRemoveDataset } from '../api/client'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  // Session state
  const [provider,   setProvider]   = useState('')
  const [mainModel,  setMainModel]  = useState('')
  const [fastModel,  setFastModel]  = useState('')
  const [isReady,    setIsReady]    = useState(false)

  // Data state
  const [dataSource, setDataSource] = useState('Synthetic E-Commerce Dataset')
  const [uploadedFile, setUploadedFile] = useState(null)
  const [datasets, setDatasets] = useState([])
  const [activeDataset, setActiveDataset] = useState(null)
  const [activeVersion, setActiveVersion] = useState('original')
  const [hasModified, setHasModified] = useState(false)
  const [dataLabTable, setDataLabTable] = useState(null)

  const _isModified = (fn) => fn === 'modified.csv'

  const addDataset = useCallback((datasetInfo) => {
    setDatasets((prev) => {
      const exists = prev.findIndex(d => d.filename === datasetInfo.filename)
      if (exists >= 0) {
        const updated = [...prev]
        updated[exists] = datasetInfo
        return updated
      }
      return [...prev, datasetInfo]
    })
    if (!_isModified(datasetInfo.filename)) {
      setActiveDataset(datasetInfo.filename)
    }
  }, [])

  const removeDataset = useCallback(async (filename) => {
    try {
      await apiRemoveDataset(filename)
      setDatasets(prev => {
        const remaining = prev.filter(d => d.filename !== filename)
        if (_isModified(filename) || !remaining.some(d => _isModified(d.filename))) {
          setHasModified(false)
          setActiveVersion('original')
        }
        const realRemaining = remaining.filter(d => !_isModified(d.filename))
        if (realRemaining.length === 0) {
          setActiveDataset(null)
          setHasModified(false)
          setActiveVersion('original')
        } else if (activeDataset === filename) {
          setActiveDataset(realRemaining[0]?.filename || remaining[0]?.filename || null)
        }
        return remaining
      })
      setMessages(prev => prev.filter(m => m.dataset !== filename))
      setQueryHistory(prev => prev.filter(m => m.dataset !== filename))
      return { success: true }
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || 'Remove failed'
      return { success: false, error: msg }
    }
  }, [activeDataset])

  // ── Messages & query history ─────────────────────────────────────────────
  const [messages, setMessages] = useState([])
  const [queryHistory, setQueryHistory] = useState([])

  // ── Persisted dashboard results (survive navigation) ─────────────────────
  const [anomalyResult,     setAnomalyResult]     = useState(null)
  const [rcaResult,         setRcaResult]         = useState(null)
  const [anomalyAgentCharts,setAnomalyAgentCharts] = useState([])
  const [rcaAgentCharts,    setRcaAgentCharts]    = useState([])

  // ── Persisted chat histories (survive navigation, reset on dataset change) ─
  const ANOMALY_WELCOME = { role: 'system', text: 'Ask me anything about these anomalies — I can query the data, explain specific rows, compare distributions, or generate charts.' }
  const RCA_WELCOME     = { role: 'system', text: 'Ask me anything about the root causes — I can traverse the causal graph, run SQL, compare time periods, or explain statistical evidence.' }
  const [anomalyChatMessages, setAnomalyChatMessages] = useState([ANOMALY_WELCOME])
  const [rcaChatMessages,     setRcaChatMessages]     = useState([RCA_WELCOME])

  // Clear persisted results when active dataset changes (stale data guard)
  useEffect(() => {
    setAnomalyResult(null)
    setRcaResult(null)
    setAnomalyAgentCharts([])
    setRcaAgentCharts([])
    setAnomalyChatMessages([ANOMALY_WELCOME])
    setRcaChatMessages([RCA_WELCOME])
  }, [activeDataset]) // eslint-disable-line

  // ── Analysis history for Insights Hub ────────────────────────────────────
  // Entries: {id, timestamp, type, query, charts, insights, dataset, version, sql}
  const [analysisHistory, setAnalysisHistory] = useState([])

  const addToAnalysisHistory = useCallback((entry) => {
    setAnalysisHistory(prev => [
      { ...entry, id: Date.now() + Math.random(), timestamp: new Date().toISOString() },
      ...prev,
    ].slice(0, 300))
  }, [])

  // ── Memory ────────────────────────────────────────────────────────────────
  const [memoryStats, setMemoryStats] = useState({
    l2: { count: 0, pct: 0 },
    l3: { nodes: 0, edges: 0 },
    l4: { count: 0, pct: 0 },
  })

  // ── UI state ──────────────────────────────────────────────────────────────
  const [activePage, setActivePage] = useState('intelligence')
  const [memoryModalLayer, setMemoryModalLayer] = useState(null)
  const [isQuerying, setIsQuerying] = useState(false)

  const refreshMemory = useCallback(async () => {
    if (!isReady) return
    try {
      const stats = await getMemoryStats()
      setMemoryStats(stats)
    } catch (_) {}
  }, [isReady])

  useEffect(() => {
    if (!isReady) return
    refreshMemory()
    const id = setInterval(refreshMemory, 30_000)
    return () => clearInterval(id)
  }, [isReady, refreshMemory])

  // Fetch existing datasets from backend on app ready (handles auto-reconnect)
  useEffect(() => {
    if (!isReady) return
    getDatasets()
      .then((list) => {
        if (Array.isArray(list) && list.length > 0) {
          list.forEach((ds) => addDataset(ds))
        }
      })
      .catch(() => {})
  }, [isReady]) // eslint-disable-line

  const addMessage = useCallback((msg) => {
    const newMsg = { id: Date.now() + Math.random(), timestamp: new Date().toISOString(), ...msg }
    setMessages((prev) => [...prev, newMsg])
    // Track assistant result messages in query history + analysis history
    if (msg.role === 'assistant' && msg.result) {
      const histEntry = {
        ...msg.result,
        query:     msg.query     || '',
        timestamp: newMsg.timestamp,
        id:        newMsg.id,
        dataset:   msg.dataset   || null,
        version:   msg.version   || 'original',
      }
      setQueryHistory((prev) => [histEntry, ...prev].slice(0, 100))

      // Also push to analysisHistory if there are charts or insights
      if ((msg.result.charts?.length > 0) || msg.result.insights) {
        addToAnalysisHistory({
          type:     msg.result.intent || 'analysis',
          query:    msg.query || '',
          charts:   msg.result.charts || [],
          insights: msg.result.insights || '',
          sql:      msg.result.sql     || '',
          dataset:  msg.dataset  || null,
          version:  msg.version  || 'original',
        })
      }
    }
  }, [addToAnalysisHistory])

  const clearChat = useCallback(() => {
    setMessages([])
    setQueryHistory([])
  }, [])

  const openDataLab = useCallback((table = null) => {
    setDataLabTable(table)
    setActivePage('datalab')
  }, [])

  const value = {
    provider, setProvider,
    mainModel, setMainModel,
    fastModel, setFastModel,
    isReady, setIsReady,
    dataSource, setDataSource,
    uploadedFile, setUploadedFile,
    datasets, addDataset, removeDataset,
    activeDataset, setActiveDataset,
    activeVersion, setActiveVersion,
    hasModified, setHasModified,
    dataLabTable, setDataLabTable,
    openDataLab,
    messages, addMessage, clearChat,
    queryHistory,
    // Persisted dashboard results
    anomalyResult,     setAnomalyResult,
    rcaResult,         setRcaResult,
    anomalyAgentCharts, setAnomalyAgentCharts,
    rcaAgentCharts,     setRcaAgentCharts,
    // Persisted chat histories
    anomalyChatMessages, setAnomalyChatMessages,
    rcaChatMessages,     setRcaChatMessages,
    // Analysis history
    analysisHistory, addToAnalysisHistory,
    memoryStats, refreshMemory,
    activePage, setActivePage,
    memoryModalLayer, setMemoryModalLayer,
    isQuerying, setIsQuerying,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export const useApp = () => {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
