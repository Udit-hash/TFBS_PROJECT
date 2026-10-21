import { useState } from 'react';
import './App.css';

const NUCLEOTIDE_THEMES = {
  A: { bg: 'rgba(16, 185, 129, 0.15)', border: '#10b981', text: '#34d399', bar: '#10b981' },
  C: { bg: 'rgba(59, 130, 246, 0.15)', border: '#3b82f6', text: '#60a5fa', bar: '#3b82f6' },
  G: { bg: 'rgba(245, 158, 11, 0.15)', border: '#f59e0b', text: '#fbbf24', bar: '#f59e0b' },
  T: { bg: 'rgba(239, 68, 68, 0.15)', border: '#ef4444', text: '#f87171', bar: '#ef4444' }
};

const MOTIF_POOL = [
  { name: 'TATA_BOX', seq: 'TATAAA' },
  { name: 'GC_BOX', seq: 'GGGCGG' },
  { name: 'CAAT_BOX', seq: 'CCAAT' }
];

export default function App() {
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFile, setUploadedFile] = useState('');
  const [shapeMatrix, setShapeMatrix] = useState(null);
  const [shapeLoading, setShapeLoading] = useState(false);
  const [shapeError, setShapeError] = useState(null);

  const getRandomBases = (len) => {
    const bases = ['A', 'C', 'G', 'T'];
    let res = '';
    for (let i = 0; i < len; i++) res += bases[Math.floor(Math.random() * bases.length)];
    return res;
  };

  const loadPreset = (type) => {
    // 1. Immediately wipe old results so old saliency scores don't misalign with the new sequence
    setResult(null);
    setError(null);
    setUploadedFile('');
    setShapeMatrix(null);
    setShapeError(null);

    if (type === 'positive') {
      const selected = MOTIF_POOL[Math.floor(Math.random() * MOTIF_POOL.length)];
      let seqArr = getRandomBases(101).split('');
      const insertPos = 35;
      for (let i = 0; i < selected.seq.length; i++) {
        seqArr[insertPos + i] = selected.seq[i];
      }
      setSequence(seqArr.join(''));
    } else {
      setSequence(getRandomBases(101));
    }
  };

  const handleSequenceFile = async (event) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = '';
    if (!file) return;

    setError(null);
    setResult(null);
    setUploadedFile('');
    setShapeMatrix(null);
    setShapeError(null);

    if (!/\.(fa|fasta|fna|txt)$/i.test(file.name)) {
      setError('Choose a FASTA or text file containing one DNA sequence.');
      return;
    }

    try {
      const contents = await file.text();
      const lines = contents.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      const headers = lines.filter((line) => line.startsWith('>'));
      if (headers.length > 1) {
        throw new Error('Upload one FASTA record at a time.');
      }

      const importedSequence = lines
        .filter((line) => !line.startsWith('>'))
        .join('')
        .toUpperCase();

      if (!importedSequence) {
        throw new Error('The selected file does not contain a DNA sequence.');
      }
      if (!/^[ACGT]+$/.test(importedSequence)) {
        throw new Error('The uploaded sequence must contain only A, C, G, and T.');
      }

      setSequence(importedSequence);
      setUploadedFile(file.name);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to read the selected file.';
      setError(message);
    }
  };

  const handlePredict = async () => {
    setError(null);
    if (sequence.length !== 101) {
      setError(`Sequence length must be exactly 101 bp. Current length: ${sequence.length} bp.`);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Inference failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to contact the ML API.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleShapeMatrix = async () => {
    setShapeError(null);
    setShapeMatrix(null);
    if (sequence.length !== 101) {
      setShapeError(`Sequence length must be exactly 101 bp. Current length: ${sequence.length} bp.`);
      return;
    }

    setShapeLoading(true);
    try {
      const res = await fetch('/api/dna-shape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `DNA-shape prediction failed (${res.status})`);
      }

      setShapeMatrix(await res.json());
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to generate the DNA-shape matrix.';
      setShapeError(message);
    } finally {
      setShapeLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <span className="eyebrow">Multi-Architecture Regulatory Genomics Framework (ICAIBE)</span>
          <h1>TFBS Prediction & Interpretability Workbench</h1>
        </div>
        <div className="pipeline-tag">
          Pipeline: <strong>{result?.model || 'CNN + BiLSTM + Transformer'}</strong>
        </div>
      </header>

      <main className="workspace">
        <section className="panel input-panel">
          <div className="panel-header">
            <label>Genomic Sequence Input (101 bp)</label>
            <div className="preset-actions">
              <label className="preset-btn upload-btn">
                Upload FASTA
                <input
                  type="file"
                  accept=".fa,.fasta,.fna,.txt,text/plain"
                  onChange={handleSequenceFile}
                />
              </label>
              <button className="preset-btn" onClick={() => loadPreset('positive')}>
                + Generate Random Motif Sequence
              </button>
              <button className="preset-btn ghost" onClick={() => loadPreset('negative')}>
                + Control Sequence
              </button>
            </div>
          </div>

          <textarea
            className="sequence-input"
            value={sequence}
            onChange={(e) => {
              setSequence(e.target.value.toUpperCase().replace(/[^ATCG]/g, ''));
              setUploadedFile('');
              setShapeMatrix(null);
              setShapeError(null);
              if (result) setResult(null);
            }}
            rows={3}
            placeholder="Paste raw 101 bp DNA sequence..."
          />
          <div className="input-hint">
            {uploadedFile
              ? `Loaded ${uploadedFile}. You can generate a DNAshapeR feature matrix for this sequence.`
              : 'Upload a single FASTA record (.fa, .fasta, .fna) or paste a 101 bp sequence.'}
          </div>

          <div className="action-row">
            <span className={`length-badge ${sequence.length === 101 ? 'ok' : 'warn'}`}>
              Sequence Length: {sequence.length} / 101 bp
            </span>

            <button
              className="primary-button"
              onClick={handlePredict}
              disabled={loading || sequence.length !== 101}
            >
              {loading ? 'Executing Tri-Branch Pipeline...' : 'Run Binding Inference'}
            </button>
            <button
              className="secondary-button"
              onClick={handleShapeMatrix}
              disabled={shapeLoading || sequence.length !== 101}
            >
              {shapeLoading ? 'Generating shape matrix...' : 'Generate DNA shape matrix'}
            </button>
          </div>

          {error && <div className="alert error">{error}</div>}
          {shapeError && <div className="alert error">{shapeError}</div>}
        </section>

        {shapeMatrix && (
          <section className="panel analysis-panel">
            <div className="panel-header tight">
              <h3>DNA Shape Feature Matrix</h3>
              <span>{shapeMatrix.predictor} {shapeMatrix.predictor_version}</span>
            </div>
            <p className="shape-description">
              Predicted structural features for the entered sequence. Each row is a native DNAshapeR position;
              blank cells indicate positions not defined for that feature.
            </p>
            <div className="table-wrap shape-matrix-scroll">
              <table className="shape-table">
                <thead>
                  <tr>
                    <th>Position</th>
                    {shapeMatrix.feature_names.map((feature) => <th key={feature}>{feature}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {shapeMatrix.rows.map((row) => (
                    <tr key={row.position}>
                      <td>{row.position}</td>
                      {shapeMatrix.feature_names.map((feature) => (
                        <td key={feature}>{row[feature] ?? '—'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {result && (
          <div className="results-stack">
            <div className="stats-grid">
              <div className="metric-card accent">
                <span className="metric-label">Binding Probability</span>
                <div className="metric-value">{(result.binding_probability * 100).toFixed(2)}%</div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${result.binding_probability * 100}%` }} />
                </div>
              </div>

              <div className="metric-card">
                <span className="metric-label">Functional Verdict</span>
                <div className="verdict-text">{result.prediction_label}</div>
                <span className={`verdict-badge ${result.prediction === 1 ? 'success' : 'neutral'}`}>
                  Threshold: 0.50
                </span>
              </div>

              <div className="metric-card">
                <span className="metric-label">Recovered Motif / JASPAR Profile</span>
                <div className="motif-name">{result.visualizations?.sequence_logo?.matched_jaspar_id}</div>
                <div className="motif-meta">
                  Core: <span>{result.visualizations?.sequence_logo?.core_sequence}</span> (Pos {result.visualizations?.sequence_logo?.window_start})
                </div>
              </div>
            </div>

            {result.visualizations?.saliency_map && (
              <div className="panel analysis-panel">
                <div className="panel-header tight">
                  <h3>1. ShiftSmooth Gradient Attribution Track</h3>
                  <span>Nucleotide Resolution Importance</span>
                </div>

                <div className="saliency-chart-wrap">
                  <div className="saliency-chart">
                    {sequence.split('').map((char, i) => {
                      const score = result.visualizations.saliency_map[i] || 0;
                      const theme = NUCLEOTIDE_THEMES[char] || NUCLEOTIDE_THEMES.A;

                      return (
                        <div key={i} className="base-col" title={`Pos ${i} (${char}): ${score}`}>
                          <div
                            className="base-bar"
                            style={{
                              height: `${Math.max(score * 70, 3)}px`,
                              backgroundColor: theme.bar
                            }}
                          />
                          <div className="base-letter" style={{ backgroundColor: theme.bg, borderColor: theme.border, color: theme.text }}>
                            {char}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {result.visualizations?.in_silico_mutagenesis?.length > 0 && (
              <div className="panel analysis-panel">
                <h3 className="table-title">2. In Silico Mutagenesis (ISM)</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Position</th>
                        <th>Mutation</th>
                        <th>Binding Score Shift (ΔP)</th>
                        <th>Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.visualizations.in_silico_mutagenesis.map((row, idx) => (
                        <tr key={idx}>
                          <td>Pos {row.position}</td>
                          <td className="mut-name">{row.mutation}</td>
                          <td className={row.delta_p < 0 ? 'delta negative' : 'delta positive'}>
                            {row.delta_p > 0 ? `+${row.delta_p}` : row.delta_p}
                          </td>
                          <td>
                            {row.delta_p < -0.1 ? 'Binding disruption (Core motif damage)' : 'Contextual flanking attenuation'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {result.visualizations?.sequence_logo && (
              <div className="panel analysis-panel">
                <h3 className="table-title">3. Learned Motif Frequency Weights</h3>
                <div className="logo-grid">
                  {result.visualizations.sequence_logo.frequency_matrix.map((col, idx) => (
                    <div key={idx} className="logo-column">
                      <span className="column-index">Pos {result.visualizations.sequence_logo.window_start + idx}</span>
                      {Object.entries(col).map(([b, freq]) => (
                        <div key={b} className="logo-entry" style={{ color: NUCLEOTIDE_THEMES[b].text }}>
                          {b}: {(freq * 100).toFixed(0)}%
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}