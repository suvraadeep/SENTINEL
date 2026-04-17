import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import OperatorGrid from '../components/auth/OperatorGrid'
import ModelSelector from '../components/auth/ModelSelector'
import ApiKeyInput from '../components/auth/ApiKeyInput'
import { getCatalogue, uploadFile } from '../api/client'
import { ChevronRight, ChevronLeft, Upload, Database, X, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

const STEPS = ['Select Provider', 'Choose Model', 'Connect API']

export default function LoginPage() {
  const navigate = useNavigate()
  const {
    provider, setProvider,
    mainModel, setMainModel,
    fastModel, setFastModel,
    setIsReady, setDataSource, setUploadedFile,
    addDataset,
  } = useApp()

  const [step, setStep] = useState(0)
  const [catalogue, setCatalogue] = useState({})
  // Multi-dataset: array of {file, progress, result, error, id}
  const [uploads, setUploads] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [loading, setCatalogueLoading] = useState(true)
  const fileInputRef = useRef(null)

  // Load provider catalogue
  useEffect(() => {
    getCatalogue()
      .then(setCatalogue)
      .catch(() => { })
      .finally(() => setCatalogueLoading(false))
  }, [])

  // Set defaults when provider changes
  useEffect(() => {
    if (provider && catalogue[provider]) {
      const cat = catalogue[provider]
      if (!mainModel) setMainModel(cat.default_main)
      if (!fastModel) setFastModel(cat.default_fast)
    }
  }, [provider, catalogue])

  // Step navigation
  const canNext = () => {
    if (step === 0) return !!provider
    if (step === 1) return !!mainModel
    return false
  }

  const handleNext = () => {
    if (canNext()) setStep((s) => s + 1)
  }

  const handleBack = () => setStep((s) => Math.max(0, s - 1))

  // Upload a single file and add to uploads list
  const uploadSingleFile = async (file) => {
    const id = Date.now() + Math.random()
    setUploads((prev) => [...prev, { id, file, progress: 0, result: null, error: null }])
    try {
      const result = await uploadFile(file, (pct) =>
        setUploads((prev) => prev.map((u) => u.id === id ? { ...u, progress: pct } : u))
      )
      setUploads((prev) => prev.map((u) => u.id === id ? { ...u, result, progress: 100 } : u))
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
      setUploads((prev) => prev.map((u) => u.id === id
        ? { ...u, error: err?.response?.data?.detail || 'Upload failed', progress: 0 }
        : u
      ))
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) uploadSingleFile(file)
    e.target.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files ?? [])
    const allowed = ['.csv', '.xlsx', '.xls', '.parquet', '.db', '.sqlite', '.sqlite3']
    const validFile = files.find(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)))
    if (validFile) uploadSingleFile(validFile)
  }

  const removeUpload = (id) => setUploads((prev) => prev.filter((u) => u.id !== id))

  // Called after successful API key validation
  const handleConnected = () => {
    setIsReady(true)
    navigate('/')
  }

  return (
    <div className="min-h-screen bg-sentinel-bg bg-grid flex flex-col items-center justify-center px-4 relative overflow-hidden">

      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-sentinel-blue/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-sentinel-cyan/5 rounded-full blur-3xl" />
      </div>

      {/* Logo */}
      <div className="mb-10 flex flex-col items-center gap-1.5 relative z-10">
        <img src="/logos/sentinels_logo.png?v=2" alt="SENTINEL" className="h-20 drop-shadow-lg" onError={(e) => { e.target.style.display = 'none' }} />
        <div className="text-center">
          <div className="text-2xl font-bold text-gradient-blue tracking-wide">SENTINEL</div>
          <div className="text-xs text-sentinel-muted tracking-widest uppercase mt-1">
            Multi-Agent Analytics Intelligence
          </div>
        </div>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-2 mb-8 relative z-10">
        {STEPS.map((label, i) => (
          <React.Fragment key={i}>
            <div className="flex items-center gap-2">
              <div className={clsx(
                'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300',
                i < step ? 'bg-sentinel-blue text-white' :
                  i === step ? 'bg-sentinel-blue text-white ring-2 ring-sentinel-blue/30' :
                    'bg-sentinel-card border border-sentinel-border text-sentinel-faint'
              )}>
                {i < step ? '✓' : i + 1}
              </div>
              <span className={clsx('text-xs font-medium hidden sm:block',
                i === step ? 'text-sentinel-text' : 'text-sentinel-faint'
              )}>
                {label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={clsx('w-12 h-px transition-all duration-300',
                i < step ? 'bg-sentinel-blue' : 'bg-sentinel-border'
              )} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Main card */}
      <div className="w-full max-w-2xl relative z-10 animate-fade-in">
        <div className="card p-8 shadow-sentinel">

          {/* Step 0 — Operator Selection */}
          {step === 0 && (
            <div>
              <h2 className="text-xl font-semibold text-sentinel-text mb-1">
                Select Intelligence Provider
              </h2>
              <p className="text-sm text-sentinel-muted mb-6">
                Choose your AI model operator. You can switch providers any time.
              </p>
              {loading ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="h-28 skeleton rounded-2xl" />
                  ))}
                </div>
              ) : (
                <OperatorGrid
                  selected={provider}
                  onSelect={(p) => { setProvider(p); setMainModel(''); setFastModel('') }}
                  catalogue={catalogue}
                />
              )}
            </div>
          )}

          {/* Step 1 — Model + Data */}
          {step === 1 && (
            <div>
              <h2 className="text-xl font-semibold text-sentinel-text mb-1">
                Model &amp; Data Configuration
              </h2>
              <p className="text-sm text-sentinel-muted mb-6">
                Select your primary model and optionally upload your own dataset.
              </p>

              <ModelSelector
                provider={provider}
                catalogue={catalogue}
                mainModel={mainModel}
                fastModel={fastModel}
                onMainChange={setMainModel}
                onFastChange={setFastModel}
              />

              {/* Multi-dataset upload section */}
              <div className="mt-6 pt-6 border-t border-sentinel-border">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="text-sm font-medium text-sentinel-text">Dataset</div>
                    <div className="text-xs text-sentinel-muted mt-0.5">
                      Upload a CSV, Excel, Parquet, or SQLite file (up to 500 MB)
                    </div>
                  </div>
                </div>

                {/* Uploaded files list */}
                {uploads.length > 0 && (
                  <div className="space-y-2 mb-3">
                    {uploads.map((u) => (
                      <div key={u.id} className={clsx(
                        'flex items-center gap-3 p-3 rounded-lg border transition-colors',
                        u.error
                          ? 'bg-sentinel-red/5 border-sentinel-red/20'
                          : u.result
                            ? 'bg-sentinel-green/5 border-sentinel-green/20'
                            : 'bg-sentinel-card border-sentinel-border'
                      )}>
                        {u.result ? (
                          <CheckCircle2 className="w-4 h-4 text-sentinel-green flex-shrink-0" />
                        ) : u.error ? (
                          <AlertCircle className="w-4 h-4 text-sentinel-red flex-shrink-0" />
                        ) : (
                          <Loader2 className="w-4 h-4 text-sentinel-blue flex-shrink-0 animate-spin" />
                        )}
                        <div className="flex-1 min-w-0">
                          <div className={clsx('text-sm font-medium truncate', u.error ? 'text-sentinel-red' : u.result ? 'text-sentinel-green' : 'text-sentinel-text')}>
                            {u.file.name}
                          </div>
                          {u.result && (
                            <div className="text-xs text-sentinel-muted">
                              {u.result.row_count?.toLocaleString()} rows · {u.result.tables?.length} table(s)
                              {u.result.dataset_count > 1 && (
                                <span className="ml-2 text-sentinel-cyan">· {u.result.dataset_count} datasets loaded</span>
                              )}
                            </div>
                          )}
                          {u.error && <div className="text-xs text-sentinel-red">{u.error}</div>}
                          {!u.result && !u.error && u.progress > 0 && (
                            <div className="mt-1.5">
                              <div className="progress-bar">
                                <div className="progress-fill bg-sentinel-blue" style={{ width: `${u.progress}%` }} />
                              </div>
                            </div>
                          )}
                        </div>
                        <button onClick={() => removeUpload(u.id)} className="flex-shrink-0 p-1 text-sentinel-faint hover:text-sentinel-text transition-colors">
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Drop zone — always visible so user can add more */}
                <div
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={clsx(
                    'flex flex-col items-center gap-2 p-5 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 group',
                    isDragging
                      ? 'border-sentinel-blue bg-sentinel-blue/5 scale-[1.01]'
                      : 'border-sentinel-border hover:border-sentinel-blue/50 hover:bg-sentinel-card/50'
                  )}
                >
                  <Upload className={clsx('w-7 h-7 transition-colors', isDragging ? 'text-sentinel-blue' : 'text-sentinel-faint group-hover:text-sentinel-blue')} />
                  <div className="text-sm text-sentinel-muted group-hover:text-sentinel-text text-center transition-colors">
                    {uploads.length === 0
                      ? <>Drop your dataset here or <span className="text-sentinel-blue">browse</span></>
                      : <>Replace dataset — drag & drop or <span className="text-sentinel-blue">browse</span></>
                    }
                  </div>
                  {uploads.length === 0 && (
                    <div className="text-xs text-sentinel-faint">
                      Or continue with the built-in e-commerce dataset
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept=".csv,.xlsx,.xls,.parquet,.db,.sqlite,.sqlite3"
                    onChange={handleFileChange}
                  />
                </div>


              </div>
            </div>
          )}

          {/* Step 2 — API Key */}
          {step === 2 && (
            <ApiKeyInput
              provider={provider}
              mainModel={mainModel}
              fastModel={fastModel}
              onConnected={handleConnected}
            />
          )}

          {/* Navigation */}
          <div className={clsx('flex mt-8', step === 0 ? 'justify-end' : 'justify-between')}>
            {step > 0 && (
              <button onClick={handleBack} className="btn-secondary">
                <ChevronLeft className="w-4 h-4" /> Back
              </button>
            )}
            {step < 2 && (
              <button
                onClick={handleNext}
                disabled={!canNext()}
                className="btn-primary"
              >
                Next <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <p className="mt-8 text-xs text-sentinel-faint relative z-10">
        SENTINEL Analytics Platform · Enterprise AI Intelligence
      </p>
    </div>
  )
}
