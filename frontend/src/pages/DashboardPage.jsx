import { useState, useRef } from 'react'
import { useApp } from '../context/AppContext'
import Sidebar from '../components/layout/Sidebar'
import TopBar from '../components/layout/TopBar'
import ChatWindow from '../components/chat/ChatWindow'
import QueryInput from '../components/chat/QueryInput'
import MemoryModal from '../components/memory/MemoryModal'
import DataLabPage from '../components/datalab/DataLabPanel'
import AnalyticsPage from './AnalyticsPage'
import AnomalyDashboard from './AnomalyDashboard'
import RCADashboard from './RCADashboard'
import { uploadFile } from '../api/client'
import { Upload, Database, Loader2, CheckCircle2 } from 'lucide-react'
import clsx from 'clsx'

// ── Re-upload overlay (Page-2 theme, blurred backdrop) ───────────────────────
function ReUploadOverlay() {
  const { setDataSource, setUploadedFile, addDataset } = useApp()
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleUpload = async (file) => {
    setUploading(true)
    setError(null)
    try {
      const result = await uploadFile(file)
      setUploadResult(result)
      setUploadedFile(file)
      setDataSource(file.name.replace(/\.[^.]+$/, ''))
      addDataset({
        filename: file.name,
        tables: result.tables ?? [],
        row_count: result.row_count ?? 0,
        date_min: result.date_min,
        date_max: result.date_max,
      })
    } catch (err) {
      setError(err?.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
    e.target.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files ?? [])
    const allowed = ['.csv', '.xlsx', '.xls', '.parquet', '.db', '.sqlite', '.sqlite3']
    const validFile = files.find(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)))
    if (validFile) handleUpload(validFile)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        backdropFilter: 'blur(12px)',
        background: 'rgba(10, 12, 20, 0.85)',
        animation: 'overlayFadeIn 0.4s ease-out',
      }}>
      <style>{`
        @keyframes overlayFadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes cardSlideUp {
          from { opacity: 0; transform: translateY(30px) scale(0.95); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
      <div className="w-full max-w-lg p-8 rounded-2xl border-2 border-sentinel-border bg-sentinel-surface/95 shadow-2xl"
        style={{
          boxShadow: '0 0 60px rgba(99, 102, 241, 0.08), 0 0 30px rgba(6, 182, 212, 0.06)',
          animation: 'cardSlideUp 0.5s ease-out 0.1s both',
        }}>

        <div className="flex items-center gap-3 mb-2">
          <Database className="w-6 h-6 text-sentinel-cyan" />
          <h2 className="text-xl font-bold text-sentinel-text">No Dataset Loaded</h2>
        </div>
        <p className="text-sm text-sentinel-muted mb-6">
          All datasets have been removed. Upload a new dataset to continue analysis.
        </p>

        {/* Upload result */}
        {uploadResult && (
          <div className="flex items-center gap-2 mb-4 p-3 rounded-lg bg-sentinel-green/5 border border-sentinel-green/20">
            <CheckCircle2 className="w-4 h-4 text-sentinel-green flex-shrink-0" />
            <span className="text-sm text-sentinel-green">
              {uploadResult.row_count?.toLocaleString()} rows · {uploadResult.tables?.length} table(s) loaded
            </span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-sentinel-red/5 border border-sentinel-red/20">
            <span className="text-sm text-sentinel-red">{error}</span>
          </div>
        )}

        {/* Drop zone — same theme as page 2 */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => !uploading && fileInputRef.current?.click()}
          className={clsx(
            'flex flex-col items-center gap-3 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 group',
            uploading
              ? 'border-sentinel-blue bg-sentinel-blue/5 pointer-events-none'
              : isDragging
                ? 'border-sentinel-cyan bg-sentinel-cyan/5 scale-[1.01]'
                : 'border-sentinel-border hover:border-sentinel-blue/50 hover:bg-sentinel-card/50'
          )}
        >
          {uploading ? (
            <>
              <Loader2 className="w-10 h-10 text-sentinel-blue animate-spin" />
              <div className="text-sm text-sentinel-muted">Processing dataset...</div>
            </>
          ) : (
            <>
              <Upload className={clsx('w-10 h-10 transition-colors',
                isDragging ? 'text-sentinel-cyan' : 'text-sentinel-faint group-hover:text-sentinel-blue')} />
              <div className="text-sm text-sentinel-muted group-hover:text-sentinel-text text-center transition-colors">
                <span className="font-medium text-sentinel-text">Click to browse</span>{' '}
                or drag & drop your dataset
              </div>
              <div className="text-xs text-sentinel-faint">
                CSV, Excel, Parquet, SQLite — up to 500 MB
              </div>
            </>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls,.parquet,.db,.sqlite,.sqlite3"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>
    </div>
  )
}


export default function DashboardPage() {
  const { memoryModalLayer, setMemoryModalLayer, activePage, datasets } = useApp()

  const isIntelligence = activePage === 'intelligence'
  const isDataLab = activePage === 'datalab'

  // Check if there are any real (non-.modified) datasets
  const realDatasets = datasets.filter(d => d.filename !== 'modified.csv')
  const noDatasets = realDatasets.length === 0

  return (
    <div className="flex h-screen bg-sentinel-bg overflow-hidden">
      {/* Left sidebar */}
      <Sidebar />

      {/* Main content area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* TopBar is hidden for DataLab — DataLab has its own header */}
        {!isDataLab && <TopBar />}

        {/* Page content */}
        {isIntelligence ? (
          <div className="flex-1 overflow-hidden flex flex-col">
            <ChatWindow />
            <QueryInput />
          </div>
        ) : activePage === 'analytics' || activePage === 'signals' || activePage === 'archives' ? (
          <AnalyticsPage />
        ) : activePage === 'anomaly' ? (
          <AnomalyDashboard />
        ) : activePage === 'rca' ? (
          <RCADashboard />
        ) : isDataLab ? (
          // DataLab fills the full remaining area (no TopBar, no chat)
          <DataLabPage />
        ) : (
          <div className="flex-1 overflow-hidden flex flex-col">
            <ChatWindow />
            <QueryInput />
          </div>
        )}
      </div>

      {/* Memory slide-over modal */}
      {memoryModalLayer && (
        <MemoryModal
          layer={memoryModalLayer}
          onClose={() => setMemoryModalLayer(null)}
          datasets={datasets}
        />
      )}

      {/* Re-upload overlay — shown when ALL datasets are removed */}
      {noDatasets && <ReUploadOverlay />}
    </div>
  )
}
