import React, { useState } from 'react';

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

  const handlePredict = async () => {
    setError(null);
    if (sequence.length !== 101) {
      setError(`Sequence length must be exactly 101 bp. Current length: ${sequence.length} bp.`);
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Inference failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', padding: '40px 24px', maxWidth: '1200px', margin: '0 auto', color: '#e2e8f0', fontFamily: 'sans-serif' }}>
      
      {/* Header */}
      <header style={{ borderBottom: '1px solid #1e293b', paddingBottom: '20px', marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '2px', color: '#38bdf8', fontFamily: 'monospace' }}>
            Multi-Architecture Regulatory Genomics Framework (ICAIBE)
          </span>
          <h1 style={{ fontSize: '26px', fontWeight: '700', color: '#f8fafc', marginTop: '6px' }}>
            TFBS Prediction & Interpretability Workbench
          </h1>
        </div>
        <div style={{ textAlign: 'right', fontSize: '12px', color: '#94a3b8', fontFamily: 'monospace' }}>
          Pipeline: <strong style={{ color: '#38bdf8' }}>CNN + BiLSTM + Transformer</strong>
        </div>
      </header>

      {/* Input Sequence Section */}
      <section style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '24px', marginBottom: '32px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
          <label style={{ fontSize: '12px', fontFamily: 'monospace', textTransform: 'uppercase', color: '#94a3b8' }}>
            Genomic Sequence Input (101 bp)
          </label>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button onClick={() => loadPreset('positive')} style={{ background: 'none', border: 'none', color: '#38bdf8', fontSize: '12px', cursor: 'pointer', fontFamily: 'monospace' }}>
              + Generate Random Motif Sequence
            </button>
            <button onClick={() => loadPreset('negative')} style={{ background: 'none', border: 'none', color: '#64748b', fontSize: '12px', cursor: 'pointer', fontFamily: 'monospace' }}>
              + Control Sequence
            </button>
          </div>
        </div>

        <textarea
          value={sequence}
          onChange={(e) => {setSequence(e.target.value.toUpperCase().replace(/[^ATCG]/g, ''));
          if (result) setResult(null);
          }} // Clear stale saliency map when sequence is modified
          rows={3}
          placeholder="Paste raw 101 bp DNA sequence..."
          style={{
            width: '100%',
            backgroundColor: '#090d16',
            border: '1px solid #1e293b',
            borderRadius: '8px',
            padding: '14px',
            fontFamily: 'monospace',
            fontSize: '13px',
            color: '#f8fafc',
            letterSpacing: '1.5px',
            resize: 'none',
            outline: 'none'
          }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
          <span style={{ fontSize: '12px', fontFamily: 'monospace', color: sequence.length === 101 ? '#34d399' : '#f59e0b' }}>
            Sequence Length: {sequence.length} / 101 bp
          </span>

          <button
            onClick={handlePredict}
            disabled={loading || sequence.length !== 101}
            style={{
              backgroundColor: sequence.length === 101 ? '#0284c7' : '#334155',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              padding: '10px 24px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: sequence.length === 101 && !loading ? 'pointer' : 'not-allowed'
            }}
          >
            {loading ? 'Executing Tri-Branch Pipeline...' : 'Run Binding Inference'}
          </button>
        </div>

        {error && (
          <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#f87171', borderRadius: '8px', fontSize: '13px' }}>
            {error}
          </div>
        )}
      </section>

      {/* Results View */}
      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
          
          {/* Top Score Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '20px' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', fontFamily: 'monospace', color: '#94a3b8' }}>
                Binding Probability (Tri-Branch Fused)
              </span>
              <div style={{ fontSize: '36px', fontWeight: '800', color: '#38bdf8', fontFamily: 'monospace', marginTop: '8px' }}>
                {(result.binding_probability * 100).toFixed(2)}%
              </div>
              <div style={{ width: '100%', height: '6px', backgroundColor: '#090d16', borderRadius: '3px', marginTop: '12px', overflow: 'hidden' }}>
                <div style={{ width: `${result.binding_probability * 100}%`, height: '100%', backgroundColor: '#38bdf8', transition: 'width 0.6s ease' }} />
              </div>
            </div>

            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '20px' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', fontFamily: 'monospace', color: '#94a3b8' }}>
                Functional Verdict
              </span>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#ffffff', marginTop: '10px' }}>
                {result.prediction_label}
              </div>
              <span style={{
                display: 'inline-block',
                marginTop: '12px',
                fontSize: '11px',
                fontFamily: 'monospace',
                padding: '3px 8px',
                borderRadius: '4px',
                backgroundColor: result.prediction === 1 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(100, 116, 139, 0.2)',
                color: result.prediction === 1 ? '#34d399' : '#94a3b8',
                border: `1px solid ${result.prediction === 1 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(100, 116, 139, 0.3)'}`
              }}>
                Threshold: 0.50
              </span>
            </div>

            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '20px' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', fontFamily: 'monospace', color: '#94a3b8' }}>
                Recovered Motif / JASPAR Profile
              </span>
              <div style={{ marginTop: '10px', fontSize: '15px', color: '#38bdf8', fontFamily: 'monospace', fontWeight: '600' }}>
                {result.visualizations?.sequence_logo?.matched_jaspar_id}
              </div>
              <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '6px', fontFamily: 'monospace' }}>
                Core: <span style={{ color: '#f8fafc' }}>{result.visualizations?.sequence_logo?.core_sequence}</span> (Pos {result.visualizations?.sequence_logo?.window_start})
              </div>
            </div>
          </div>

          {/* Module 1: ShiftSmooth Saliency Track */}
          {result.visualizations?.saliency_map && (
            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#f8fafc', fontFamily: 'monospace' }}>
                  1. ShiftSmooth Gradient Attribution Track (Averaged Over Cyclic Shifts)
                </h3>
                <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>
                  Nucleotide Resolution Importance
                </span>
              </div>

              <div style={{ overflowX: 'auto', paddingBottom: '12px', paddingTop: '32px' }}>
                <div style={{ display: 'inline-flex', gap: '3px', alignItems: 'flex-end', minWidth: '100%' }}>
                  {sequence.split('').map((char, i) => {
                    const score = result.visualizations.saliency_map[i] || 0;
                    const theme = NUCLEOTIDE_THEMES[char] || NUCLEOTIDE_THEMES.A;
                    return (
                      <div key={i} title={`Pos ${i} (${char}): ${score}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                        <div
                          style={{
                            width: '12px',
                            backgroundColor: theme.bar,
                            height: `${Math.max(score * 70, 3)}px`,
                            borderTopLeftRadius: '2px',
                            borderTopRightRadius: '2px',
                            opacity: 0.85
                          }}
                        />
                        <div
                          style={{
                            width: '16px',
                            height: '20px',
                            marginTop: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '10px',
                            fontWeight: '700',
                            fontFamily: 'monospace',
                            borderRadius: '3px',
                            backgroundColor: theme.bg,
                            border: `1px solid ${theme.border}`,
                            color: theme.text
                          }}
                        >
                          {char}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Module 2: In Silico Mutagenesis Profile */}
          {result.visualizations?.in_silico_mutagenesis?.length > 0 && (
            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '24px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#f8fafc', fontFamily: 'monospace', marginBottom: '14px' }}>
                2. In Silico Mutagenesis (ISM) – Single-Nucleotide Variant Impacts
              </h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #1e293b', color: '#94a3b8', fontFamily: 'monospace' }}>
                    <th style={{ padding: '8px' }}>Position</th>
                    <th style={{ padding: '8px' }}>Mutation</th>
                    <th style={{ padding: '8px' }}>Binding Score Shift (ΔP)</th>
                    <th style={{ padding: '8px' }}>Mechanistic Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {result.visualizations.in_silico_mutagenesis.map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '10px 8px', fontFamily: 'monospace' }}>Pos {row.position}</td>
                      <td style={{ padding: '10px 8px', fontFamily: 'monospace', color: '#38bdf8' }}>{row.mutation}</td>
                      <td style={{ padding: '10px 8px', fontFamily: 'monospace', color: row.delta_p < 0 ? '#f87171' : '#34d399' }}>
                        {row.delta_p > 0 ? `+${row.delta_p}` : row.delta_p}
                      </td>
                      <td style={{ padding: '10px 8px' }}>
                        {row.delta_p < -0.1 ? 'Binding disruption (Core motif damage)' : 'Contextual flanking attenuation'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Module 3: Sequence Logo Matrix */}
          {result.visualizations?.sequence_logo && (
            <div style={{ backgroundColor: '#131b2e', border: '1px solid #1e293b', borderRadius: '14px', padding: '24px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: '600', color: '#f8fafc', fontFamily: 'monospace', marginBottom: '14px' }}>
                3. Learned Motif Sequence Frequency Weights (PFM Representation)
              </h3>
              <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                {result.visualizations.sequence_logo.frequency_matrix.map((col, idx) => (
                  <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '4px', background: '#090d16', padding: '10px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                    <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>Pos {result.visualizations.sequence_logo.window_start + idx}</span>
                    {Object.entries(col).map(([b, freq]) => (
                      <div key={b} style={{ fontSize: '12px', fontFamily: 'monospace', color: NUCLEOTIDE_THEMES[b].text }}>
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

    </div>
  );
}