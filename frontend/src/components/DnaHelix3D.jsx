import { useEffect, useMemo, useRef, useState } from 'react';
import { HelixRenderer, BASE_COLORS, COMPLEMENT, HBONDS, heatColor, rgbToCss } from '../lib/helixRenderer';
import { C, mono, panelStyle, sectionTitle, chip, iconButton } from './ui';

const WINDOW_SIZES = [
  { label: '12 bp', value: 12 },
  { label: '24 bp', value: 24 },
  { label: '48 bp', value: 48 },
  { label: 'All', value: 101 },
];

const BASE_NAMES = { A: 'Adenine', C: 'Cytosine', G: 'Guanine', T: 'Thymine' };

export default function DnaHelix3D({ sequence, result, onAsk }) {
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);
  const viz = result?.visualizations || {};
  const saliency = useMemo(() => viz.saliency_map || [], [viz.saliency_map]);
  const ismEffect = useMemo(() => viz.ism_position_effect || [], [viz.ism_position_effect]);
  const logo = viz.sequence_logo;
  const coreStart = logo?.window_start ?? null;
  const coreLen = logo?.core_sequence?.length || 0;
  const hasIsm = ismEffect.length === sequence.length;
  const ismNorm = useMemo(() => {
    const worst = Math.max(0, ...ismEffect.map((d) => -d));
    return ismEffect.map((d) => (worst > 0 && d < 0 ? -d / worst : 0));
  }, [ismEffect]);

  const [colorMode, setColorMode] = useState('importance');
  // Phones get a shorter window so the helix stays readable
  const initialLen = typeof window !== 'undefined' && window.innerWidth < 640 ? 12 : 24;
  const [windowLen, setWindowLen] = useState(initialLen);
  const [windowStart, setWindowStart] = useState(() =>
    coreStart === null ? 0 : Math.max(0, coreStart + Math.floor(coreLen / 2) - Math.floor(initialLen / 2))
  );
  const [selected, setSelected] = useState(null);
  const [zoom, setZoom] = useState(1);

  // Create the renderer once.
  useEffect(() => {
    const r = new HelixRenderer(canvasRef.current, { onSelect: setSelected, background: C.canvas });
    rendererRef.current = r;
    return () => r.destroy();
  }, []);

  useEffect(() => {
    rendererRef.current?.setData({ sequence, saliency, ismEffect, coreStart, coreLen });
  }, [sequence, saliency, ismEffect, coreStart, coreLen]);

  useEffect(() => { rendererRef.current?.setWindow(windowStart, windowLen); }, [windowStart, windowLen, sequence]);
  useEffect(() => { rendererRef.current?.setColorMode(colorMode); }, [colorMode]);
  useEffect(() => { rendererRef.current?.setSelected(selected); }, [selected]);
  useEffect(() => { rendererRef.current?.setZoom(zoom); }, [zoom]);

  const maxStart = Math.max(0, sequence.length - windowLen);
  const centreOn = (pos) => setWindowStart(Math.max(0, Math.min(maxStart, pos - Math.floor(windowLen / 2))));

  // Rank = how many positions score at least as high (ties count against, matching the tutor)
  const rank = useMemo(() => saliency.map((s) => saliency.filter((v) => v >= s).length), [saliency]);

  const selInfo = selected === null ? null : (() => {
    const base = sequence[selected];
    const comp = COMPLEMENT[base];
    const muts = (viz.in_silico_mutagenesis || []).filter((m) => m.position === selected);
    return {
      base, comp,
      hb: HBONDS[base],
      score: saliency[selected] ?? 0,
      rank: rank[selected],
      ism: hasIsm ? ismEffect[selected] : null,
      muts,
      inCore: coreStart !== null && selected >= coreStart && selected < coreStart + coreLen,
    };
  })();

  return (
    <div style={panelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px', flexWrap: 'wrap', marginBottom: '14px' }}>
        <div>
          <h3 style={sectionTitle}>3D DNA view: where the model looks</h3>
          <p style={{ fontSize: '13px', color: C.muted, marginTop: '6px', maxWidth: '640px', lineHeight: 1.5 }}>
            Your sequence as a double helix. Each rung is a base pair, and the dashed lines are its hydrogen bonds
            (A–T has 2, G–C has 3). Drag to rotate. Click a base pair for details.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
          <button style={chip(colorMode === 'importance')} onClick={() => setColorMode('importance')}>Model importance</button>
          <button
            style={{ ...chip(colorMode === 'ism'), opacity: hasIsm ? 1 : 0.4 }}
            disabled={!hasIsm}
            title={hasIsm ? '' : 'Restart the backend to get per-position mutation data'}
            onClick={() => setColorMode('ism')}
          >Mutation impact</button>
          <button style={chip(colorMode === 'bases')} onClick={() => setColorMode('bases')}>Bases</button>
        </div>
      </div>

      <div style={{ position: 'relative', background: C.canvas, borderRadius: '10px', border: `1px solid ${C.border}`, overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          aria-label="3D double helix of the input sequence coloured by model importance"
          style={{ width: '100%', height: 'clamp(260px, 62vw, 380px)', display: 'block', cursor: 'grab', touchAction: 'none' }}
        />
        <div style={{ position: 'absolute', top: '10px', right: '10px', display: 'flex', gap: '6px' }}>
          <button style={iconButton} title="Rotate left" onClick={() => rendererRef.current?.rotateBy(-0.5)}>⟲</button>
          <button style={iconButton} title="Rotate right" onClick={() => rendererRef.current?.rotateBy(0.5)}>⟳</button>
          <button style={iconButton} title="Zoom out" onClick={() => setZoom((z) => Math.max(0.6, z / 1.25))}>−</button>
          <button style={iconButton} title="Zoom in" onClick={() => setZoom((z) => Math.min(3, z * 1.25))}>+</button>
          <button style={iconButton} title="Reset view" onClick={() => { setZoom(1); rendererRef.current?.resetView(); }}>Reset</button>
        </div>
      </div>
      <Legend mode={colorMode} />

      {/* Navigator: the full 101 bp with the visible window highlighted */}
      <div style={{ marginTop: '14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '11px', fontFamily: mono, color: C.muted, textTransform: 'uppercase', letterSpacing: '1px' }}>
            Showing positions {windowStart}–{Math.min(sequence.length, windowStart + windowLen) - 1} · click the strip to move
          </span>
          <div style={{ display: 'flex', gap: '6px' }}>
            {WINDOW_SIZES.map((w) => (
              <button key={w.value} style={chip(windowLen === w.value)} onClick={() => {
                setWindowLen(w.value);
                setWindowStart((s) => Math.max(0, Math.min(s, sequence.length - w.value)));
              }}>{w.label}</button>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '1px', alignItems: 'flex-end', height: '34px', cursor: 'pointer' }} role="group" aria-label="Sequence navigator">
          {sequence.split('').map((b, i) => {
            const inWin = i >= windowStart && i < windowStart + windowLen;
            const v = colorMode === 'ism' ? ismNorm[i] || 0 : saliency[i] || 0;
            return (
              <div
                key={i}
                title={`Position ${i} (${b})`}
                onClick={() => { centreOn(i); setSelected(i); }}
                style={{
                  flex: 1,
                  height: `${Math.max(12, v * 100)}%`,
                  background: colorMode === 'bases' ? BASE_COLORS[b] : rgbToCss(heatColor(v)),
                  opacity: inWin ? 1 : 0.28,
                  outline: selected === i ? '2px solid #fff' : 'none',
                  borderRadius: '1px',
                }}
              />
            );
          })}
        </div>
      </div>

      {selInfo && (
        <div style={{ marginTop: '16px', display: 'flex', gap: '18px', alignItems: 'center', flexWrap: 'wrap', background: C.panelDeep, border: `1px solid ${C.border}`, borderRadius: '10px', padding: '14px 16px' }}>
          <div style={{ fontFamily: mono, fontSize: '22px', fontWeight: 700 }}>
            <span style={{ color: BASE_COLORS[selInfo.base] }}>{selInfo.base}</span>
            <span style={{ color: C.faint }}> – </span>
            <span style={{ color: BASE_COLORS[selInfo.comp] }}>{selInfo.comp}</span>
          </div>
          <div style={{ fontSize: '13px', color: C.body, lineHeight: 1.6, flex: 1, minWidth: '240px' }}>
            <div><strong>Position {selected}</strong>: {BASE_NAMES[selInfo.base]} paired with {BASE_NAMES[selInfo.comp]}, held by <strong>{selInfo.hb} hydrogen bonds</strong>{selInfo.inCore ? ' · inside the peak window' : ''}</div>
            <div style={{ color: C.muted }}>
              Model importance {selInfo.score.toFixed(2)} (#{selInfo.rank} of {sequence.length})
              {selInfo.ism !== null && selInfo.ism < 0 && <> · worst single mutation here lowers the score by {(Math.abs(selInfo.ism) * 100).toFixed(1)} points</>}
              {selInfo.ism !== null && selInfo.ism >= 0 && <> · no mutation here lowers the score</>}
            </div>
          </div>
          {onAsk && (
            <button
              style={{ ...chip(true), padding: '8px 14px' }}
              onClick={() => onAsk(`Why is position ${selected} (${selInfo.base}) important for my prediction?`)}
            >Ask the tutor about this ↗</button>
          )}
        </div>
      )}

      <p style={{ fontSize: '11px', color: C.faint, marginTop: '12px', fontFamily: mono }}>
        Idealised B-DNA geometry (10.5 bp/turn). The model reads only the letters; colours show its attributions, not measured atomic contacts.
      </p>
    </div>
  );
}

function Legend({ mode }) {
  const box = { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '4px 12px', marginTop: '10px', fontSize: '11px', fontFamily: mono, color: C.muted };
  if (mode === 'bases') {
    return (
      <div style={box}>
        {Object.entries(BASE_COLORS).map(([b, col]) => (
          <span key={b} style={{ marginRight: '10px' }}><span style={{ color: col, fontWeight: 700 }}>■</span> {b}</span>
        ))}
        <span>┄ hydrogen bond</span>
      </div>
    );
  }
  const stops = [0, 0.35, 0.7, 1].map((v) => rgbToCss(heatColor(v))).join(',');
  return (
    <div style={box}>
      <span>{mode === 'ism' ? 'Score drop if mutated:' : 'ShiftSmooth importance:'}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span>low</span>
        <span style={{ width: '110px', height: '8px', borderRadius: '4px', background: `linear-gradient(90deg, ${stops})` }} />
        <span>high</span>
      </span>
      <span>┄ dashed lines = hydrogen bonds (A–T 2, G–C 3)</span>
    </div>
  );
}
