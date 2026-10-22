import { useEffect, useRef, useState } from 'react';
import { C, mono, panelStyle, sectionTitle, chip } from './ui';

// Experimentally solved protein–DNA complexes from the RCSB Protein Data Bank.
export const STRUCTURES = [
  {
    id: '1CDW', tf: 'TBP', site: 'TATA box', match: (r) => hasMotif(r, 'TATA_BOX') || /TBP|TATA/.test(r.core),
    blurb: 'TATA-binding protein grabs the TATA box from the minor groove and bends the DNA sharply, like a saddle sitting on it.',
  },
  {
    id: '1AAY', tf: 'Zif268 (EGR1)', site: 'GC-rich site', match: (r) => hasMotif(r, 'GC_BOX') || /SP1/.test(r.jaspar),
    blurb: 'A three-zinc-finger protein from the same family as SP1. Each "finger" reads 3 bases in the major groove. It is shown because it is the classic, well-resolved example of how SP1-like proteins read GC-rich DNA.',
  },
  {
    id: '4AWL', tf: 'NF-Y', site: 'CCAAT box', match: (r) => hasMotif(r, 'CAAT_BOX') || /CCAAT/.test(r.core),
    blurb: 'A three-part complex. Two subunits wrap the DNA like a histone, and the third reaches into the minor groove to read CCAAT.',
  },
  {
    id: '1DH3', tf: 'CREB', site: 'CRE (TGACGTCA)', match: (r) => /TGACG|CGTCA/.test(r.core),
    blurb: 'Two CREB proteins form a pair. Their long helices insert into the major groove like chopsticks gripping the DNA.',
  },
  {
    id: '5KKQ', tf: 'CTCF', site: 'CTCF core site', match: (r) => /CCCTC|GAGGG/.test(r.core),
    blurb: 'Five of CTCF’s eleven zinc fingers wrap around the DNA, following the major groove.',
  },
];

const DNA_RESN = ['DA', 'DC', 'DG', 'DT'];
const DNA_COLORS = { DA: '#10b981', DC: '#3b82f6', DG: '#f59e0b', DT: '#ef4444' };
const VIEWER_SRC = 'https://3Dmol.org/build/3Dmol-min.js';

function hasMotif(r, name) {
  return (r.motifs || []).includes(name);
}

export function pickStructure(result) {
  const viz = result?.visualizations || {};
  const info = {
    motifs: result?.motifs_detected || [],
    core: viz.sequence_logo?.core_sequence || '',
    jaspar: viz.sequence_logo?.matched_jaspar_id || '',
  };
  return STRUCTURES.find((s) => s.match(info)) || null;
}

// Load the 3Dmol.js viewer only when this tab is first opened (keeps the app light).
let viewerLibPromise = null;
function load3Dmol() {
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (!viewerLibPromise) {
    viewerLibPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = VIEWER_SRC;
      s.async = true;
      s.onload = () => (window.$3Dmol ? resolve(window.$3Dmol) : reject(new Error('3Dmol did not initialise')));
      s.onerror = () => { viewerLibPromise = null; reject(new Error('Could not load the 3D viewer (are you online?)')); };
      document.head.appendChild(s);
    });
  }
  return viewerLibPromise;
}

export default function StructureViewer({ result, onAsk }) {
  const suggested = pickStructure(result);
  const [current, setCurrent] = useState(suggested || STRUCTURES[0]);
  const [showContacts, setShowContacts] = useState(true);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState(null);
  const hostRef = useRef(null);
  const viewerRef = useRef(null);
  const pdbCache = useRef({});

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    setError(null);

    (async () => {
      try {
        const $3Dmol = await load3Dmol();
        if (cancelled) return;
        if (!viewerRef.current) {
          viewerRef.current = $3Dmol.createViewer(hostRef.current, { backgroundColor: C.canvas, antialias: true });
        }
        let pdb = pdbCache.current[current.id];
        if (!pdb) {
          const res = await fetch(`https://files.rcsb.org/download/${current.id}.pdb`);
          if (!res.ok) throw new Error(`Could not download ${current.id} from the PDB`);
          pdb = await res.text();
          pdbCache.current[current.id] = pdb;
        }
        if (cancelled) return;

        const v = viewerRef.current;
        v.clear();
        v.addModel(pdb, 'pdb');
        v.setStyle({}, { cartoon: { color: '#a78bfa', opacity: 0.9 } });
        v.setStyle({ resn: 'HOH' }, {});
        DNA_RESN.forEach((resn) => {
          v.setStyle({ resn }, { stick: { color: DNA_COLORS[resn], radius: 0.22 } });
        });
        if (showContacts) {
          try {
            // Protein residues with any atom within 3.5 Å of the DNA: likely contacts / hydrogen bonds
            v.addStyle(
              { within: { distance: 3.5, sel: { resn: DNA_RESN } }, not: { resn: [...DNA_RESN, 'HOH'] }, byres: true },
              { stick: { color: '#fde047', radius: 0.18 } }
            );
          } catch { /* older 3Dmol versions: skip the contact highlight */ }
        }
        v.zoomTo({ resn: DNA_RESN });
        v.render();
        setStatus('ready');
      } catch (e) {
        if (!cancelled) { setStatus('error'); setError(e.message); }
      }
    })();

    return () => { cancelled = true; };
  }, [current, showContacts]);

  useEffect(() => () => { viewerRef.current?.clear(); viewerRef.current = null; }, []);

  return (
    <div style={panelStyle}>
      <h3 style={sectionTitle}>Real protein–DNA structure</h3>
      <p style={{ fontSize: '13px', color: C.muted, marginTop: '6px', lineHeight: 1.5, maxWidth: '760px' }}>
        A 3D structure measured in the lab (X-ray crystallography), showing what binding physically looks like.
        {suggested
          ? <> Your peak window resembles a <strong style={{ color: C.text }}>{suggested.site}</strong>, so the matching complex is loaded.</>
          : <> Your sequence didn’t match one of these known sites, so pick any example below.</>}
      </p>

      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', margin: '14px 0' }}>
        {STRUCTURES.map((s) => (
          <button key={s.id} style={chip(current.id === s.id)} onClick={() => setCurrent(s)}>
            {s.tf} · {s.site}{suggested?.id === s.id ? ' ★' : ''}
          </button>
        ))}
      </div>

      <div style={{ position: 'relative', height: '420px', borderRadius: '10px', border: `1px solid ${C.border}`, overflow: 'hidden', background: C.canvas }}>
        <div ref={hostRef} style={{ position: 'absolute', inset: 0 }} />
        {status !== 'ready' && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '10px', color: C.muted, fontFamily: mono, fontSize: '13px', textAlign: 'center', padding: '20px' }}>
            {status === 'loading' ? `Loading PDB ${current.id}…` : (
              <>
                <span style={{ color: C.danger }}>{error}</span>
                <a href={`https://www.rcsb.org/3d-view/${current.id}`} target="_blank" rel="noreferrer" style={{ color: C.accent }}>
                  Open {current.id} on rcsb.org instead ↗
                </a>
              </>
            )}
          </div>
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap', marginTop: '14px', alignItems: 'flex-start' }}>
        <div style={{ fontSize: '13px', color: C.body, lineHeight: 1.6, maxWidth: '700px' }}>
          <strong>{current.tf}</strong> (PDB <a href={`https://www.rcsb.org/structure/${current.id}`} target="_blank" rel="noreferrer" style={{ color: C.accent }}>{current.id}</a>): {current.blurb}
          <div style={{ fontFamily: mono, fontSize: '11px', color: C.muted, marginTop: '8px' }}>
            <span style={{ color: '#a78bfa' }}>■</span> protein&nbsp;&nbsp;
            <span style={{ color: DNA_COLORS.DA }}>■</span> A&nbsp;
            <span style={{ color: DNA_COLORS.DC }}>■</span> C&nbsp;
            <span style={{ color: DNA_COLORS.DG }}>■</span> G&nbsp;
            <span style={{ color: DNA_COLORS.DT }}>■</span> T&nbsp;&nbsp;
            {showContacts && <><span style={{ color: '#fde047' }}>■</span> protein residues touching DNA (≤ 3.5 Å)</>}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button style={chip(showContacts)} onClick={() => setShowContacts((s) => !s)}>Highlight contacts</button>
          {onAsk && (
            <button style={chip(true)} onClick={() => onAsk(`How does ${current.tf} physically recognise its DNA site?`)}>
              Ask the tutor ↗
            </button>
          )}
        </div>
      </div>
      <p style={{ fontSize: '11px', color: C.faint, marginTop: '12px', fontFamily: mono }}>
        This is a reference structure for the matching protein family, not a simulation of your sequence. Drag to rotate, scroll to zoom.
      </p>
    </div>
  );
}
