import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 300_000,   // 5 min — LLM queries can take time
  headers: { 'Content-Type': 'application/json' },
})

// ── Provider / Auth ───────────────────────────────────────────────────────
export const configureProvider = (payload) =>
  api.post('/api/provider/configure', payload).then((r) => r.data)

export const getModels = (provider) =>
  api.get('/api/provider/models', { params: { provider } }).then((r) => r.data)

export const getCatalogue = () =>
  api.get('/api/provider/catalogue').then((r) => r.data)

// ── Query ─────────────────────────────────────────────────────────────────
export const runQuery = (query, dataset = null) =>
  api.post('/api/query', { query, ...(dataset ? { dataset } : {}) }).then((r) => r.data)

// ── Upload ────────────────────────────────────────────────────────────────
export const uploadFile = (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress?.(Math.round((e.loaded / e.total) * 100)),
  }).then((r) => r.data)
}

// ── Memory ────────────────────────────────────────────────────────────────
export const getMemoryStats  = () => api.get('/api/memory/stats').then((r) => r.data)
export const getMemoryLayer  = (layer, dataset = null) =>
  api.get(`/api/memory/layer/${layer}`, { params: dataset ? { dataset } : {} }).then((r) => r.data)

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = () => api.get('/api/health').then((r) => r.data)

// ── DataLab ───────────────────────────────────────────────────────────────
export const getDatasets      = ()             => api.get('/api/datalab/datasets').then((r) => r.data)
export const removeDataset    = (filename)     => api.delete(`/api/datalab/datasets/${encodeURIComponent(filename)}`).then((r) => r.data)
export const getDataLabTables = ()             => api.get('/api/datalab/tables').then((r) => r.data)
export const getTableSchema   = (table)        => api.get(`/api/datalab/schema/${table}`).then((r) => r.data)
export const identifyDataset  = (prompt)       => api.post('/api/datalab/identify-dataset', { prompt }).then((r) => r.data)
export const previewTable     = (table, limit = 100, offset = 0) =>
  api.get(`/api/datalab/preview/${table}`, { params: { limit, offset } }).then((r) => r.data)
export const executeOperation  = (payload)    => api.post('/api/datalab/execute', payload).then((r) => r.data)
export const runDataLabSql     = (sql)        => api.post('/api/datalab/sql', { sql }).then((r) => r.data)
export const transformData     = (payload)    => api.post('/api/datalab/transform', payload).then((r) => r.data)
export const autoPlotTable     = (table, sql = '') =>
  api.get(`/api/datalab/autoplot/${table}`, { params: sql ? { sql } : {} }).then((r) => r.data)
export const requestCustomPlot = (payload)    => api.post('/api/datalab/plot', payload).then((r) => r.data)
export const downloadTable     = (table)      => `${API_BASE}/api/datalab/download/${table}`
export const querySchema       = (table, prompt) => api.post(`/api/datalab/schema/${encodeURIComponent(table)}/query`, { prompt }).then((r) => r.data)
export const promoteDataset    = (payload)    => api.post('/api/datalab/promote', payload).then((r) => r.data)
export const dropTable         = (table)      => api.delete(`/api/datalab/tables/${encodeURIComponent(table)}`).then((r) => r.data)
export const switchVersion     = (version)    => api.post('/api/datalab/switch-version', { version }).then((r) => r.data)

// ── Anomaly ───────────────────────────────────────────────────────────────
export const detectAnomalies  = (payload)    => api.post('/api/anomaly/detect', payload).then((r) => r.data)
// payload: { message, context?, table?, chat_history? }
export const anomalyChat      = (payload)    => api.post('/api/anomaly/chat', payload).then((r) => r.data)

// ── RCA ───────────────────────────────────────────────────────────────────
export const runRCA       = (payload)  => api.post('/api/rca/analyze',  payload).then((r) => r.data)
// payload: { message, context?, table?, chat_history? }
export const rcaChat      = (payload)  => api.post('/api/rca/chat',     payload).then((r) => r.data)
export const rcaTraverse  = (payload)  => api.post('/api/rca/traverse', payload).then((r) => r.data)

