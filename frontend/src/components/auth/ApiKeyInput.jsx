import React, { useState } from 'react'
import { Lock, Eye, EyeOff, CheckCircle, XCircle, Loader2, Shield } from 'lucide-react'
import { configureProvider } from '../../api/client'

const PROVIDER_LABELS = {
  openai:    { name: 'OpenAI', placeholder: 'sk-proj-...',   hint: 'Find at platform.openai.com/api-keys' },
  google:    { name: 'Google', placeholder: 'AIza...',        hint: 'Find at console.cloud.google.com' },
  anthropic: { name: 'Anthropic', placeholder: 'sk-ant-...', hint: 'Find at console.anthropic.com/keys' },
  groq:      { name: 'Groq',   placeholder: 'gsk_...',       hint: 'Find at console.groq.com/keys' },
  nvidia:    { name: 'NVIDIA', placeholder: 'nvapi-...',     hint: 'Find at integrate.api.nvidia.com' },
}

export default function ApiKeyInput({ provider, mainModel, fastModel, onConnected }) {
  const [apiKey, setApiKey] = useState('')
  const [visible, setVisible] = useState(false)
  const [status, setStatus] = useState('idle')   // idle | loading | valid | invalid
  const [error, setError] = useState('')

  const info = PROVIDER_LABELS[provider] || { name: provider, placeholder: 'your-api-key', hint: '' }

  const handleValidate = async () => {
    if (!apiKey.trim()) return
    setStatus('loading')
    setError('')

    try {
      const res = await configureProvider({
        provider,
        api_key: apiKey.trim(),
        main_model: mainModel,
        fast_model: fastModel,
      })
      if (res.valid) {
        setStatus('valid')
        // Small delay to show the valid state before navigating
        setTimeout(onConnected, 800)
      } else {
        setStatus('invalid')
        setError(res.error || 'Invalid API key')
      }
    } catch (err) {
      setStatus('invalid')
      setError(err?.response?.data?.detail || 'Connection failed. Is the backend running?')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleValidate()
  }

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col items-center gap-3 mb-8">
        <div className="w-14 h-14 rounded-2xl bg-sentinel-blue/10 border border-sentinel-blue/20 flex items-center justify-center">
          <Lock className="w-7 h-7 text-sentinel-blue" />
        </div>
        <div className="text-center">
          <h2 className="text-xl font-semibold text-sentinel-text">Secure Authorization</h2>
          <p className="text-sm text-sentinel-muted mt-1">
            Enter your {info.name} API key to connect SENTINEL to your model operator
          </p>
        </div>
      </div>

      {/* Key input */}
      <div className="space-y-2 mb-4">
        <label className="section-label">
          {info.name} API Key
        </label>
        <div className="relative">
          <input
            type={visible ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => { setApiKey(e.target.value); setStatus('idle'); setError('') }}
            onKeyDown={handleKeyDown}
            placeholder={info.placeholder}
            className="input pr-12 font-mono text-sm"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            onClick={() => setVisible((v) => !v)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-sentinel-faint hover:text-sentinel-muted"
            tabIndex={-1}
          >
            {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
        {info.hint && (
          <p className="text-xs text-sentinel-faint">{info.hint}</p>
        )}
      </div>

      {/* Validation status */}
      {status === 'valid' && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-sentinel-green/10 border border-sentinel-green/30 mb-4">
          <CheckCircle className="w-5 h-5 text-sentinel-green flex-shrink-0" />
          <div>
            <div className="text-sm font-semibold text-sentinel-green">VALID KEY DETECTED</div>
            <div className="text-xs text-sentinel-muted">Establishing secure connection...</div>
          </div>
        </div>
      )}

      {status === 'invalid' && (
        <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-sentinel-red/10 border border-sentinel-red/30 mb-4">
          <XCircle className="w-5 h-5 text-sentinel-red flex-shrink-0" />
          <div className="text-sm text-sentinel-red">{error}</div>
        </div>
      )}

      {/* Privacy note */}
      <div className="flex items-start gap-2 px-4 py-3 rounded-xl bg-sentinel-blue/5 border border-sentinel-blue/10 mb-6">
        <Shield className="w-4 h-4 text-sentinel-blue flex-shrink-0 mt-0.5" />
        <p className="text-xs text-sentinel-muted">
          Keys are stored locally within your secure session and are never transmitted to SENTINEL's
          servers. Your credentials remain under your control at all times.
        </p>
      </div>

      {/* Connect button */}
      <button
        onClick={handleValidate}
        disabled={!apiKey.trim() || status === 'loading' || status === 'valid'}
        className="btn-primary w-full justify-center py-3 text-base"
      >
        {status === 'loading' ? (
          <><Loader2 className="w-5 h-5 animate-spin" /> Validating...</>
        ) : status === 'valid' ? (
          <><CheckCircle className="w-5 h-5" /> Connected!</>
        ) : (
          <><Lock className="w-5 h-5" /> Establish Connection</>
        )}
      </button>
    </div>
  )
}
