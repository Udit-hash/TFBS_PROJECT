import { useState, useCallback } from "react";

const KNOWN_MOTIFS = { TATA_BOX: "TATAAA", GC_BOX: "GGGCGG", CAAT_BOX: "CCAAT" };

const BASE_COLORS = {
  A: "#4FD67A",
  T: "#FF6B6B",
  C: "#5B9DFF",
  G: "#FFC857",
};

function randomSeq(len) {
  const bases = ["A", "T", "C", "G"];
  let s = "";
  for (let i = 0; i < len; i++) s += bases[Math.floor(Math.random() * 4)];
  return s;
}

function SequencePreview({ sequence }) {
  if (!sequence) {
    return <span className="placeholder">sequence preview will render here</span>;
  }
  return (
    <>
      {sequence.split("").map((ch, i) => (
        <span
          key={i}
          className={BASE_COLORS[ch] ? "" : "b-invalid"}
          style={BASE_COLORS[ch] ? { color: BASE_COLORS[ch] } : undefined}
        >
          {ch}
        </span>
      ))}
    </>
  );
}

function Gauge({ probability }) {
  const pct = Math.round(probability * 1000) / 10;
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - probability);
  const color = probability >= 0.8 ? "#7DE0E6" : probability >= 0.5 ? "#FFC857" : "#7C8798";

  return (
    <div className="gauge">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle cx="70" cy="70" r={radius} fill="none" stroke="#232B38" strokeWidth="10" />
        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
          className="gauge-arc"
        />
        <text x="70" y="65" textAnchor="middle" className="gauge-pct" fill={color}>
          {pct}%
        </text>
        <text x="70" y="84" textAnchor="middle" className="gauge-label">
          binding
        </text>
      </svg>
    </div>
  );
}

function SaliencyTrack({ saliency, sequence }) {
  return (
    <div className="track-wrap">
      <div className="track-label">saliency track</div>
      <div className="track">
        {saliency.map((v, i) => {
          const ch = sequence[i] || "N";
          const color = BASE_COLORS[ch] || "#4A5364";
          const height = Math.max(3, Math.round(v * 74));
          const opacity = 0.3 + v * 0.7;
          return (
            <div
              key={i}
              className="track-bar"
              title={`${ch} pos ${i + 1}: ${v}`}
              style={{
                height: `${height}px`,
                background: color,
                opacity,
                animationDelay: `${Math.min(i * 4, 600)}ms`,
              }}
            />
          );
        })}
      </div>
      <div className="ruler">
        <span>1</span>
        <span>{sequence.length}</span>
      </div>
    </div>
  );
}

export default function App() {
  const [sequence, setSequence] = useState("");
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const insertExample = (type) => {
    const prefix = randomSeq(10 + Math.floor(Math.random() * 10));
    const suffix = randomSeq(10 + Math.floor(Math.random() * 10));
    if (type === "tata") setSequence(prefix + KNOWN_MOTIFS.TATA_BOX + suffix);
    else if (type === "gc") setSequence(prefix + KNOWN_MOTIFS.GC_BOX + suffix);
    else if (type === "caat") setSequence(prefix + KNOWN_MOTIFS.CAAT_BOX + suffix);
    else setSequence(randomSeq(30));
    setError("");
    setResult(null);
  };

  const runPrediction = useCallback(async () => {
    const seq = sequence.trim().toUpperCase();
    setError("");

    if (!seq) {
      setError("enter a sequence first");
      return;
    }
    if (!/^[ATCG]+$/.test(seq)) {
      setError("invalid sequence — only A, T, C, G allowed");
      return;
    }

    setLoading(true);
    setResult(null);
    const base = apiUrl.trim().replace(/\/$/, "");

    try {
      const res = await fetch(`${base}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: seq }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `request failed with status ${res.status}`);
      }
      const data = await res.json();
      setResult({ ...data, sequence: seq });
    } catch (e) {
      setError(`error: ${e.message} — check the API is running at ${base}`);
    } finally {
      setLoading(false);
    }
  }, [sequence, apiUrl]);

  return (
    <div className="tfbs-app">
      <style>{`
        .tfbs-app{
          --bg:#0B0E14; --panel:#131922; --panel-2:#0F1420; --border:#232B38;
          --text:#E8ECF1; --muted:#7C8798; --faint:#4A5364; --accent:#7DE0E6;
          --accent-dim:rgba(125,224,230,0.12);
          background:var(--bg); color:var(--text); min-height:100vh;
          font-family:'IBM Plex Sans', system-ui, sans-serif;
          padding:40px 24px 80px;
        }
        .tfbs-app *{box-sizing:border-box;}
        .tfbs-wrap{max-width:1080px;margin:0 auto;}
        .tfbs-header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
          margin-bottom:32px;border-bottom:1px solid var(--border);padding-bottom:20px;}
        .tfbs-header h1{font-family:'Space Grotesk', sans-serif;font-weight:700;font-size:26px;
          margin:0 0 4px;letter-spacing:-0.01em;
          background:linear-gradient(90deg,#E8ECF1,#7DE0E6);-webkit-background-clip:text;
          background-clip:text;-webkit-text-fill-color:transparent;}
        .tfbs-header .sub{color:var(--muted);font-size:13px;font-family:'IBM Plex Mono',monospace;}
        .legend{display:flex;gap:14px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);}
        .legend span{display:inline-flex;align-items:center;gap:5px;}
        .dot{width:8px;height:8px;border-radius:2px;display:inline-block;}
        .tfbs-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
        @media(max-width:820px){.tfbs-grid{grid-template-columns:1fr;}}
        .tfbs-panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;
          padding:20px;position:relative;overflow:hidden;}
        .tfbs-panel::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
          background:linear-gradient(90deg,#4FD67A,#FFC857,#FF6B6B,#5B9DFF);opacity:0.6;}
        .tfbs-panel h2{font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:500;
          text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin:0 0 14px;}
        textarea{width:100%;min-height:110px;background:var(--panel-2);border:1px solid var(--border);
          border-radius:4px;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:14px;
          padding:12px;resize:vertical;letter-spacing:0.02em;line-height:1.6;transition:border-color .15s;}
        textarea:focus{outline:none;border-color:var(--accent);
          box-shadow:0 0 0 3px var(--accent-dim);}
        .preview{margin-top:10px;min-height:44px;background:var(--panel-2);border:1px solid var(--border);
          border-radius:4px;padding:10px 12px;font-family:'IBM Plex Mono',monospace;font-size:14px;
          letter-spacing:0.03em;word-break:break-all;line-height:1.7;}
        .placeholder{color:var(--faint);}
        .b-invalid{color:var(--muted);text-decoration:underline wavy #FF6B6B;}
        .examples{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
        .ex-btn{background:transparent;border:1px solid var(--border);color:var(--muted);
          font-family:'IBM Plex Mono',monospace;font-size:11px;padding:6px 10px;border-radius:4px;
          cursor:pointer;transition:border-color .15s, color .15s, transform .1s;}
        .ex-btn:hover{border-color:var(--accent);color:var(--accent);}
        .ex-btn:active{transform:scale(0.96);}
        .tfbs-row{display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap;}
        .run-btn{background:linear-gradient(135deg,#7DE0E6,#5B9DFF);color:#06181A;border:none;
          font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:14px;padding:10px 22px;
          border-radius:4px;cursor:pointer;letter-spacing:0.02em;transition:opacity .15s, transform .1s;}
        .run-btn:hover:not(:disabled){opacity:0.9;}
        .run-btn:active:not(:disabled){transform:scale(0.97);}
        .run-btn:disabled{opacity:0.4;cursor:default;}
        .api-field{display:flex;align-items:center;gap:6px;margin-left:auto;
          font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--faint);}
        .api-field input{background:var(--panel-2);border:1px solid var(--border);color:var(--muted);
          font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 8px;border-radius:3px;width:170px;}
        .msg{margin-top:10px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#FF6B6B;min-height:16px;}
        .scan-line{position:relative;height:2px;background:var(--panel-2);margin-top:12px;
          border-radius:2px;overflow:hidden;}
        .scan-line::after{content:"";position:absolute;top:0;left:0;height:100%;width:30%;
          background:var(--accent);animation:scan 1.1s ease-in-out infinite;}
        @keyframes scan{0%{left:-30%;}100%{left:100%;}}
        .empty-state{color:var(--faint);font-family:'IBM Plex Mono',monospace;font-size:12px;
          padding:30px 0;text-align:center;}
        .results-body{animation:fadeIn .35s ease;}
        @keyframes fadeIn{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:translateY(0);}}
        .gauge-row{display:flex;align-items:center;gap:24px;flex-wrap:wrap;}
        .gauge-arc{transition:stroke-dashoffset .8s ease, stroke .3s ease;}
        .gauge-pct{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;}
        .gauge-label{font-family:'IBM Plex Mono',monospace;font-size:9px;fill:#7C8798;
          text-transform:uppercase;letter-spacing:0.05em;}
        .meta-line{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);}
        .motif-list{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
        .motif-chip{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:5px 10px;
          border-radius:3px;background:var(--accent-dim);color:var(--accent);
          border:1px solid rgba(125,224,230,0.3);animation:popIn .3s ease backwards;}
        @keyframes popIn{from{opacity:0;transform:scale(0.85);}to{opacity:1;transform:scale(1);}}
        .no-motifs{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--faint);margin-top:14px;}
        .track-wrap{margin-top:20px;}
        .track-label{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
          margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;}
        .track{display:flex;align-items:flex-end;height:80px;gap:1px;background:var(--panel-2);
          border:1px solid var(--border);border-radius:4px;padding:6px 4px 0;overflow-x:auto;}
        .track-bar{flex:0 0 auto;width:5px;min-height:2px;border-radius:1px 1px 0 0;
          animation:growUp .3s ease backwards;}
        @keyframes growUp{from{transform:scaleY(0);}to{transform:scaleY(1);}}
        .ruler{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;
          font-size:10px;color:var(--faint);margin-top:4px;padding:0 4px;}
      `}</style>

      <div className="tfbs-wrap">
        <div className="tfbs-header">
          <div>
            <h1>TFBS prediction console</h1>
            <div className="sub">transcription factor binding site scanner — phase 1 scaffold</div>
          </div>
          <div className="legend">
            <span><span className="dot" style={{ background: BASE_COLORS.A }} />A</span>
            <span><span className="dot" style={{ background: BASE_COLORS.T }} />T</span>
            <span><span className="dot" style={{ background: BASE_COLORS.C }} />C</span>
            <span><span className="dot" style={{ background: BASE_COLORS.G }} />G</span>
          </div>
        </div>

        <div className="tfbs-grid">
          <div className="tfbs-panel">
            <h2>Sequence input</h2>
            <textarea
              placeholder="Paste a DNA sequence — A, T, C, G only"
              spellCheck={false}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
            />
            <div className="preview">
              <SequencePreview sequence={sequence.trim().toUpperCase()} />
            </div>

            <div className="examples">
              <button className="ex-btn" onClick={() => insertExample("tata")}>insert TATA box example</button>
              <button className="ex-btn" onClick={() => insertExample("gc")}>insert GC box example</button>
              <button className="ex-btn" onClick={() => insertExample("caat")}>insert CAAT box example</button>
              <button className="ex-btn" onClick={() => insertExample("random")}>random background</button>
            </div>

            <div className="tfbs-row">
              <button className="run-btn" onClick={runPrediction} disabled={loading}>
                {loading ? "scanning..." : "run prediction"}
              </button>
              <div className="api-field">
                <span>API</span>
                <input
                  value={apiUrl}
                  spellCheck={false}
                  onChange={(e) => setApiUrl(e.target.value)}
                />
              </div>
            </div>
            <div className="msg">{error}</div>
            {loading && <div className="scan-line" />}
          </div>

          <div className="tfbs-panel">
            <h2>Prediction results</h2>
            {!result && !loading && (
              <div className="empty-state">run a sequence to see binding probability and saliency</div>
            )}
            {result && (
              <div className="results-body">
                <div className="gauge-row">
                  <Gauge probability={result.binding_probability} />
                  <div className="meta-line">length: {result.input_length} bp</div>
                </div>

                {result.motifs_detected && result.motifs_detected.length > 0 ? (
                  <div className="motif-list">
                    {result.motifs_detected.map((m, i) => (
                      <span key={m} className="motif-chip" style={{ animationDelay: `${i * 80}ms` }}>
                        {m}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div className="no-motifs">no known motifs detected</div>
                )}

                <SaliencyTrack
                  saliency={result.visualizations.saliency_map}
                  sequence={result.sequence}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
