import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

interface ModelResult {
  diagnosis: string;
  confidence: number;
  probabilities: number[];
  gradcam: string | null;
  label: string;
}

interface PredictResponse {
  models: {
    efficientnet_b0: ModelResult;
    deit_tiny: ModelResult;
    resnet18: ModelResult;
  };
  class_names: string[];
}

const MODEL_ORDER = ['efficientnet_b0', 'deit_tiny', 'resnet18'] as const;
const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function diagnosisColor(d: string) {
  if (d === 'normal') return 'var(--normal)';
  if (d === 'amd') return 'var(--amd)';
  if (d === 'dr') return 'var(--dr)';
  return 'var(--accent)';
}

function diagnosisLabel(d: string) {
  if (d === 'normal') return 'Normal';
  if (d === 'amd') return 'AMD';
  if (d === 'dr') return 'Diabetic Retinopathy';
  return d.toUpperCase();
}

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="prob-row">
      <span className="prob-label">{label.toUpperCase()}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${(value * 100).toFixed(1)}%`, background: color }} />
      </div>
      <span className="prob-value">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function ModelCard({
  modelKey,
  result,
  classNames,
}: {
  modelKey: string;
  result: ModelResult;
  classNames: string[];
}) {
  const [showGradcam, setShowGradcam] = useState(true);
  const imgSrc = showGradcam && result.gradcam ? result.gradcam : undefined;

  return (
    <div className={`model-card diagnosed-${result.diagnosis}`}>
      <div className="model-card-header">
        <span className="model-name">{result.label}</span>
        {result.gradcam && (
          <button
            className={`toggle-sm ${showGradcam ? 'active' : ''}`}
            onClick={() => setShowGradcam(v => !v)}
          >
            {showGradcam ? 'Grad-CAM' : 'Original'}
          </button>
        )}
      </div>

      <div className={`card-ring diagnosed-${result.diagnosis}`}>
        <div className="card-frame">
          {imgSrc ? (
            <img src={imgSrc} alt={`${result.label} Grad-CAM`} className="card-img" />
          ) : (
            <div className="card-img-placeholder" />
          )}
        </div>
      </div>

      <span className={`dx-badge dx-${result.diagnosis}`}>
        {diagnosisLabel(result.diagnosis)}
      </span>

      <div className="confidence-line">
        <span className="conf-number">{(result.confidence * 100).toFixed(1)}</span>
        <span className="conf-unit">% confidence</span>
      </div>

      <div className="prob-bars">
        {classNames.map((cls, i) => (
          <ProbBar
            key={cls}
            label={cls}
            value={result.probabilities[i]}
            color={cls === result.diagnosis ? diagnosisColor(cls) : 'var(--border-medium)'}
          />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const processFile = useCallback((f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResponse(null);
  }, []);

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResponse(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await axios.post<PredictResponse>(`${API}/predict`, fd);
      setResponse(res.data);
    } catch {
      alert('Prediction failed. Is the backend running?\n\nStart it with:\nPYTHONPATH=src uvicorn retinai.api.server:app --reload');
    } finally {
      setLoading(false);
    }
  };

  // Drag-and-drop
  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
  };

  // Paste support
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find(i => i.type.startsWith('image/'));
      if (item) { const f = item.getAsFile(); if (f) processFile(f); }
    };
    document.addEventListener('paste', onPaste);
    return () => document.removeEventListener('paste', onPaste);
  }, [processFile]);

  return (
    <div className="app" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      <div className="bg-effects" aria-hidden>
        <div className="bg-noise" />
        <div className="bg-radial" />
      </div>

      <input type="file" id="file-input" accept="image/*" style={{ display: 'none' }}
        onChange={e => { if (e.target.files?.[0]) processFile(e.target.files[0]); }} />

      <header className="header">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 48 48" width="44" height="44" fill="none">
            <path d="M8 24c0-8.8 7.2-16 16-16s16 7.2 16 16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M14 24c0 5.5 4.5 10 10 10s10-4.5 10-10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />
            <circle cx="24" cy="24" r="3" fill="currentColor" />
          </svg>
          <h1 className="brand-name">Retin<span className="brand-hi">AI</span></h1>
        </div>
        <p className="tagline">Multi-Model Retinal Disease Screening</p>
      </header>

      <main className="main">
        {!preview ? (
          /* ---- Upload Zone ---- */
          <section className="upload-zone">
            <label htmlFor="file-input" className={`upload-area ${isDragging ? 'dragging' : ''}`}>
              <div className="upload-content">
                <svg viewBox="0 0 48 48" width="48" height="48" fill="none" stroke="currentColor"
                  strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="24" cy="24" r="20" opacity="0.25" />
                  <path d="M24 16v16M16 24h16" />
                </svg>
                <span className="upload-title">Drop retinal photograph</span>
                <span className="upload-hint">or click to browse · PNG, JPG</span>
                <span className="upload-hint">or paste with ⌘V / Ctrl+V</span>
              </div>
            </label>
          </section>
        ) : !response ? (
          /* ---- Preview + Analyze ---- */
          <section className="preview-section">
            <div className={`orig-ring ${loading ? 'scanning' : ''}`}>
              <div className="orig-frame">
                <img src={preview} alt="Retinal scan" className="orig-img" />
              </div>
            </div>
            <div className="preview-actions">
              <button onClick={handleClear} className="btn btn-ghost">Clear</button>
              <label htmlFor="file-input" className="btn btn-ghost">Change image</label>
              <button onClick={handleSubmit} disabled={loading} className="btn btn-primary">
                {loading ? <><span className="spinner" /> Analyzing…</> : 'Analyze with 3 Models'}
              </button>
            </div>
          </section>
        ) : (
          /* ---- Results Grid ---- */
          <section className="results-section">
            {/* Original image column */}
            <div className="orig-col">
              <div className="orig-ring-sm">
                <div className="orig-frame-sm">
                  <img src={preview!} alt="Original scan" className="orig-img" />
                </div>
              </div>
              <span className="col-label">Original Scan</span>
              <div className="orig-actions">
                <button onClick={handleClear} className="btn btn-ghost btn-sm">New scan</button>
                <label htmlFor="file-input" className="btn btn-ghost btn-sm">Change</label>
              </div>
            </div>

            {/* Model cards */}
            <div className="model-grid">
              {MODEL_ORDER.map(key => {
                const result = response.models[key];
                if (!result) return null;
                return (
                  <ModelCard
                    key={key}
                    modelKey={key}
                    result={result}
                    classNames={response.class_names}
                  />
                );
              })}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>For research and screening assistance only — does not replace professional diagnosis</p>
      </footer>
    </div>
  );
}
