import { useState, useCallback } from "react";


// ============================================================
// CONFIGURATION
// ============================================================

const SEQUENCE_LENGTH = 101;

const DEFAULT_API_URL = "http://localhost:8000";


// ============================================================
// KNOWN BIOLOGICAL MOTIFS
// ============================================================

const KNOWN_MOTIFS = {
  TATA_BOX: "TATAAA",
  GC_BOX: "GGGCGG",
  CAAT_BOX: "CCAAT",
};


// ============================================================
// DNA BASE COLORS
// ============================================================

const BASE_COLORS = {
  A: "#4FD67A",
  T: "#FF6B6B",
  C: "#5B9DFF",
  G: "#FFC857",
};


// ============================================================
// RANDOM DNA SEQUENCE GENERATOR
// ============================================================

function randomSeq(length) {

  const bases = ["A", "T", "C", "G"];

  let sequence = "";

  for (let i = 0; i < length; i++) {

    const randomIndex = Math.floor(
      Math.random() * bases.length
    );

    sequence += bases[randomIndex];
  }

  return sequence;
}


// ============================================================
// SEQUENCE PREVIEW
// ============================================================

function SequencePreview({ sequence }) {

  if (!sequence) {

    return (
      <span className="placeholder">
        sequence preview will render here
      </span>
    );
  }

  return (
    <>
      {sequence.split("").map((base, index) => (

        <span
          key={index}
          style={{
            color:
              BASE_COLORS[base] || "#E8ECF1"
          }}
        >
          {base}
        </span>

      ))}
    </>
  );
}


// ============================================================
// PROBABILITY GAUGE
// ============================================================

function Gauge({ probability }) {

  const percentage = Math.round(
    probability * 1000
  ) / 10;

  const radius = 54;

  const circumference =
    2 * Math.PI * radius;

  const offset =
    circumference * (1 - probability);


  let color = "#7C8798";

  if (probability >= 0.8) {

    color = "#7DE0E6";

  } else if (probability >= 0.5) {

    color = "#FFC857";
  }


  return (
    <div className="gauge">

      <svg
        width="140"
        height="140"
        viewBox="0 0 140 140"
      >

        {/* Background circle */}

        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke="#232B38"
          strokeWidth="10"
        />


        {/* Probability arc */}

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


        {/* Percentage */}

        <text
          x="70"
          y="65"
          textAnchor="middle"
          className="gauge-pct"
          fill={color}
        >
          {percentage}%
        </text>


        {/* Label */}

        <text
          x="70"
          y="84"
          textAnchor="middle"
          className="gauge-label"
        >
          binding
        </text>

      </svg>

    </div>
  );
}


// ============================================================
// CNN SALIENCY VISUALIZATION
// ============================================================

function SaliencyTrack({
  saliency,
  sequence
}) {

  if (
    !saliency ||
    saliency.length === 0
  ) {

    return null;
  }


  return (
    <div className="track-wrap">

      <div className="track-label">

        CNN gradient saliency

      </div>


      <div className="track">

        {saliency.map(
          (value, index) => {

            const base =
              sequence[index] || "N";


            const baseColor =
              BASE_COLORS[base] ||
              "#4A5364";


            const height =
              Math.max(
                3,
                Math.round(
                  value * 74
                )
              );


            const opacity =
              0.3 + value * 0.7;


            return (
              <div
                key={index}
                className="track-bar"
                title={
                  `${base} | Position ${index + 1} | Saliency ${value}`
                }
                style={{
                  height: `${height}px`,
                  background: baseColor,
                  opacity: opacity,
                  animationDelay:
                    `${Math.min(
                      index * 4,
                      600
                    )}ms`
                }}
              />
            );
          }
        )}

      </div>


      <div className="ruler">

        <span>
          Position 1
        </span>

        <span>
          Position {sequence.length}
        </span>

      </div>

    </div>
  );
}


// ============================================================
// MAIN APPLICATION
// ============================================================

export default function App() {

  // ----------------------------------------------------------
  // STATE
  // ----------------------------------------------------------

  const [
    sequence,
    setSequence
  ] = useState("");


  const [
    apiUrl,
    setApiUrl
  ] = useState(
    DEFAULT_API_URL
  );


  const [
    loading,
    setLoading
  ] = useState(false);


  const [
    error,
    setError
  ] = useState("");


  const [
    result,
    setResult
  ] = useState(null);


  // ==========================================================
  // INSERT EXAMPLE
  // ==========================================================

  const insertExample = (type) => {

    let motif = "";


    // --------------------------------------------------------
    // Select motif
    // --------------------------------------------------------

    if (type === "tata") {

      motif =
        KNOWN_MOTIFS.TATA_BOX;

    }

    else if (type === "gc") {

      motif =
        KNOWN_MOTIFS.GC_BOX;

    }

    else if (type === "caat") {

      motif =
        KNOWN_MOTIFS.CAAT_BOX;

    }


    // --------------------------------------------------------
    // Random 101 bp sequence
    // --------------------------------------------------------

    if (type === "random") {

      const randomSequence =
        randomSeq(
          SEQUENCE_LENGTH
        );


      setSequence(
        randomSequence
      );

    }


    // --------------------------------------------------------
    // 101 bp sequence containing motif
    // --------------------------------------------------------

    else {

      const remainingLength =
        SEQUENCE_LENGTH -
        motif.length;


      const prefixLength =
        Math.floor(
          remainingLength / 2
        );


      const suffixLength =
        remainingLength -
        prefixLength;


      const prefix =
        randomSeq(
          prefixLength
        );


      const suffix =
        randomSeq(
          suffixLength
        );


      const exampleSequence =
        prefix +
        motif +
        suffix;


      setSequence(
        exampleSequence
      );

    }


    setError("");

    setResult(null);
  };


  // ==========================================================
  // RUN CNN PREDICTION
  // ==========================================================

  const runPrediction = useCallback(
    async () => {

      // ------------------------------------------------------
      // Clean input
      // ------------------------------------------------------

      const seq =
        sequence
          .trim()
          .toUpperCase();


      setError("");


      // ------------------------------------------------------
      // Empty validation
      // ------------------------------------------------------

      if (!seq) {

        setError(
          "Enter a DNA sequence first."
        );

        return;
      }


      // ------------------------------------------------------
      // DNA validation
      // ------------------------------------------------------

      if (
        !/^[ATCG]+$/.test(seq)
      ) {

        setError(
          "Invalid sequence. Only A, T, C and G are allowed."
        );

        return;
      }


      // ------------------------------------------------------
      // Length validation
      // ------------------------------------------------------

      if (
        seq.length !==
        SEQUENCE_LENGTH
      ) {

        setError(
          `Sequence must be exactly ${SEQUENCE_LENGTH} bp. Current length: ${seq.length} bp.`
        );

        return;
      }


      // ------------------------------------------------------
      // Start loading
      // ------------------------------------------------------

      setLoading(true);

      setResult(null);


      // Remove trailing slash
      // from API URL

      const base =
        apiUrl
          .trim()
          .replace(
            /\/$/,
            ""
          );


      try {

        // ----------------------------------------------------
        // Send sequence to FastAPI
        // ----------------------------------------------------

        const response =
          await fetch(
            `${base}/predict`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json"
              },

              body: JSON.stringify({
                sequence: seq
              })
            }
          );


        // ----------------------------------------------------
        // Handle API error
        // ----------------------------------------------------

        if (!response.ok) {

          const errorData =
            await response
              .json()
              .catch(
                () => ({})
              );


          throw new Error(
            errorData.detail ||
            `Request failed with status ${response.status}`
          );
        }


        // ----------------------------------------------------
        // Read API response
        // ----------------------------------------------------

        const data =
          await response.json();


        // ----------------------------------------------------
        // Store result
        // ----------------------------------------------------

        setResult({
          ...data,
          sequence: seq
        });

      }


      catch (err) {

        setError(
          `Error: ${err.message}. Make sure the CNN API is running at ${base}`
        );

      }


      finally {

        setLoading(false);

      }

    },
    [
      sequence,
      apiUrl
    ]
  );


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="tfbs-app">

      {/* ================================================== */}
      {/* STYLES */}
      {/* ================================================== */}

      <style>{`

        * {
          box-sizing: border-box;
        }


        .tfbs-app {

          --bg: #0B0E14;

          --panel: #131922;

          --panel-2: #0F1420;

          --border: #232B38;

          --text: #E8ECF1;

          --muted: #7C8798;

          --faint: #4A5364;

          --accent: #7DE0E6;

          --accent-dim:
            rgba(
              125,
              224,
              230,
              0.12
            );

          background:
            var(--bg);

          color:
            var(--text);

          min-height:
            100vh;

          font-family:
            'IBM Plex Sans',
            system-ui,
            sans-serif;

          padding:
            40px 24px 80px;
        }


        .tfbs-wrap {

          max-width:
            1080px;

          margin:
            0 auto;
        }


        /* ==================================================
           HEADER
        ================================================== */

        .tfbs-header {

          display:
            flex;

          justify-content:
            space-between;

          align-items:
            flex-end;

          gap:
            24px;

          margin-bottom:
            32px;

          border-bottom:
            1px solid var(--border);

          padding-bottom:
            20px;
        }


        .tfbs-header h1 {

          font-family:
            'Space Grotesk',
            sans-serif;

          font-weight:
            700;

          font-size:
            28px;

          margin:
            0 0 6px;

          letter-spacing:
            -0.01em;

          background:
            linear-gradient(
              90deg,
              #E8ECF1,
              #7DE0E6
            );

          -webkit-background-clip:
            text;

          background-clip:
            text;

          -webkit-text-fill-color:
            transparent;
        }


        .tfbs-header .sub {

          color:
            var(--muted);

          font-size:
            13px;

          font-family:
            'IBM Plex Mono',
            monospace;
        }


        /* ==================================================
           LEGEND
        ================================================== */

        .legend {

          display:
            flex;

          gap:
            14px;

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          color:
            var(--muted);
        }


        .legend span {

          display:
            inline-flex;

          align-items:
            center;

          gap:
            5px;
        }


        .dot {

          width:
            8px;

          height:
            8px;

          border-radius:
            2px;

          display:
            inline-block;
        }


        /* ==================================================
           GRID
        ================================================== */

        .tfbs-grid {

          display:
            grid;

          grid-template-columns:
            1fr 1fr;

          gap:
            20px;
        }


        @media (max-width: 820px) {

          .tfbs-grid {

            grid-template-columns:
              1fr;
          }
        }


        /* ==================================================
           PANELS
        ================================================== */

        .tfbs-panel {

          background:
            var(--panel);

          border:
            1px solid var(--border);

          border-radius:
            8px;

          padding:
            20px;

          position:
            relative;

          overflow:
            hidden;
        }


        .tfbs-panel::before {

          content:
            "";

          position:
            absolute;

          top:
            0;

          left:
            0;

          right:
            0;

          height:
            2px;

          background:
            linear-gradient(
              90deg,
              #4FD67A,
              #FFC857,
              #FF6B6B,
              #5B9DFF
            );

          opacity:
            0.6;
        }


        .tfbs-panel h2 {

          font-family:
            'Space Grotesk',
            sans-serif;

          font-size:
            13px;

          font-weight:
            500;

          text-transform:
            uppercase;

          letter-spacing:
            0.08em;

          color:
            var(--muted);

          margin:
            0 0 14px;
        }


        /* ==================================================
           TEXTAREA
        ================================================== */

        textarea {

          width:
            100%;

          min-height:
            110px;

          background:
            var(--panel-2);

          border:
            1px solid var(--border);

          border-radius:
            4px;

          color:
            var(--text);

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            14px;

          padding:
            12px;

          resize:
            vertical;

          letter-spacing:
            0.02em;

          line-height:
            1.6;

          transition:
            border-color 0.15s;
        }


        textarea:focus {

          outline:
            none;

          border-color:
            var(--accent);

          box-shadow:
            0 0 0 3px
            var(--accent-dim);
        }


        /* ==================================================
           PREVIEW
        ================================================== */

        .preview {

          margin-top:
            10px;

          min-height:
            44px;

          background:
            var(--panel-2);

          border:
            1px solid var(--border);

          border-radius:
            4px;

          padding:
            10px 12px;

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            14px;

          letter-spacing:
            0.03em;

          word-break:
            break-all;

          line-height:
            1.7;
        }


        .placeholder {

          color:
            var(--faint);
        }


        /* ==================================================
           EXAMPLE BUTTONS
        ================================================== */

        .examples {

          display:
            flex;

          flex-wrap:
            wrap;

          gap:
            8px;

          margin-top:
            12px;
        }


        .ex-btn {

          background:
            transparent;

          border:
            1px solid var(--border);

          color:
            var(--muted);

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          padding:
            7px 10px;

          border-radius:
            4px;

          cursor:
            pointer;

          transition:
            border-color 0.15s,
            color 0.15s,
            transform 0.1s;
        }


        .ex-btn:hover {

          border-color:
            var(--accent);

          color:
            var(--accent);
        }


        .ex-btn:active {

          transform:
            scale(0.96);
        }


        /* ==================================================
           ACTION ROW
        ================================================== */

        .tfbs-row {

          display:
            flex;

          gap:
            10px;

          align-items:
            center;

          margin-top:
            16px;

          flex-wrap:
            wrap;
        }


        /* ==================================================
           RUN BUTTON
        ================================================== */

        .run-btn {

          background:
            linear-gradient(
              135deg,
              #7DE0E6,
              #5B9DFF
            );

          color:
            #06181A;

          border:
            none;

          font-family:
            'Space Grotesk',
            sans-serif;

          font-weight:
            700;

          font-size:
            14px;

          padding:
            10px 22px;

          border-radius:
            4px;

          cursor:
            pointer;

          letter-spacing:
            0.02em;

          transition:
            opacity 0.15s,
            transform 0.1s;
        }


        .run-btn:hover:not(:disabled) {

          opacity:
            0.9;
        }


        .run-btn:active:not(:disabled) {

          transform:
            scale(0.97);
        }


        .run-btn:disabled {

          opacity:
            0.4;

          cursor:
            default;
        }


        /* ==================================================
           API FIELD
        ================================================== */

        .api-field {

          display:
            flex;

          align-items:
            center;

          gap:
            6px;

          margin-left:
            auto;

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          color:
            var(--faint);
        }


        .api-field input {

          background:
            var(--panel-2);

          border:
            1px solid var(--border);

          color:
            var(--muted);

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          padding:
            5px 8px;

          border-radius:
            3px;

          width:
            180px;
        }


        /* ==================================================
           ERROR MESSAGE
        ================================================== */

        .msg {

          margin-top:
            10px;

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            12px;

          color:
            #FF6B6B;

          min-height:
            16px;

          line-height:
            1.5;
        }


        /* ==================================================
           LOADING BAR
        ================================================== */

        .scan-line {

          position:
            relative;

          height:
            2px;

          background:
            var(--panel-2);

          margin-top:
            12px;

          border-radius:
            2px;

          overflow:
            hidden;
        }


        .scan-line::after {

          content:
            "";

          position:
            absolute;

          top:
            0;

          left:
            0;

          height:
            100%;

          width:
            30%;

          background:
            var(--accent);

          animation:
            scan 1.1s
            ease-in-out
            infinite;
        }


        @keyframes scan {

          0% {
            left:
              -30%;
          }

          100% {
            left:
              100%;
          }
        }


        /* ==================================================
           EMPTY STATE
        ================================================== */

        .empty-state {

          color:
            var(--faint);

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            12px;

          padding:
            30px 0;

          text-align:
            center;

          line-height:
            1.7;
        }


        /* ==================================================
           RESULTS
        ================================================== */

        .results-body {

          animation:
            fadeIn 0.35s ease;
        }


        @keyframes fadeIn {

          from {

            opacity:
              0;

            transform:
              translateY(4px);
          }

          to {

            opacity:
              1;

            transform:
              translateY(0);
          }
        }


        .gauge-row {

          display:
            flex;

          align-items:
            center;

          gap:
            24px;

          flex-wrap:
            wrap;
        }


        /* ==================================================
           GAUGE
        ================================================== */

        .gauge-arc {

          transition:
            stroke-dashoffset 0.8s ease,
            stroke 0.3s ease;
        }


        .gauge-pct {

          font-family:
            'Space Grotesk',
            sans-serif;

          font-size:
            20px;

          font-weight:
            700;
        }


        .gauge-label {

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            9px;

          fill:
            #7C8798;

          text-transform:
            uppercase;

          letter-spacing:
            0.05em;
        }


        /* ==================================================
           RESULT METADATA
        ================================================== */

        .meta-line {

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            12px;

          color:
            var(--muted);

          margin-bottom:
            5px;
        }


        .prediction-label {

          margin-top:
            10px;

          font-family:
            'Space Grotesk',
            sans-serif;

          font-size:
            15px;

          font-weight:
            600;

          line-height:
            1.4;
        }


        .model-badge {

          display:
            inline-block;

          margin-top:
            8px;

          padding:
            4px 8px;

          border:
            1px solid
            rgba(
              125,
              224,
              230,
              0.3
            );

          border-radius:
            3px;

          color:
            var(--accent);

          background:
            var(--accent-dim);

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            10px;
        }


        /* ==================================================
           MOTIFS
        ================================================== */

        .motif-list {

          display:
            flex;

          flex-wrap:
            wrap;

          gap:
            8px;

          margin-top:
            14px;
        }


        .motif-chip {

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          padding:
            5px 10px;

          border-radius:
            3px;

          background:
            var(--accent-dim);

          color:
            var(--accent);

          border:
            1px solid
            rgba(
              125,
              224,
              230,
              0.3
            );

          animation:
            popIn 0.3s
            ease backwards;
        }


        @keyframes popIn {

          from {

            opacity:
              0;

            transform:
              scale(0.85);
          }

          to {

            opacity:
              1;

            transform:
              scale(1);
          }
        }


        .no-motifs {

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            12px;

          color:
            var(--faint);

          margin-top:
            14px;
        }


        /* ==================================================
           SALIENCY
        ================================================== */

        .track-wrap {

          margin-top:
            20px;
        }


        .track-label {

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            11px;

          color:
            var(--muted);

          margin-bottom:
            8px;

          text-transform:
            uppercase;

          letter-spacing:
            0.06em;
        }


        .track {

          display:
            flex;

          align-items:
            flex-end;

          height:
            80px;

          gap:
            1px;

          background:
            var(--panel-2);

          border:
            1px solid var(--border);

          border-radius:
            4px;

          padding:
            6px 4px 0;

          overflow-x:
            auto;
        }


        .track-bar {

          flex:
            0 0 auto;

          width:
            5px;

          min-height:
            2px;

          border-radius:
            1px 1px 0 0;

          animation:
            growUp 0.3s
            ease backwards;
        }


        @keyframes growUp {

          from {

            transform:
              scaleY(0);
          }

          to {

            transform:
              scaleY(1);
          }
        }


        .ruler {

          display:
            flex;

          justify-content:
            space-between;

          font-family:
            'IBM Plex Mono',
            monospace;

          font-size:
            10px;

          color:
            var(--faint);

          margin-top:
            4px;

          padding:
            0 4px;
        }


        /* ==================================================
           RESPONSIVE
        ================================================== */

        @media (max-width: 600px) {

          .tfbs-app {

            padding:
              24px 12px 50px;
          }


          .tfbs-header {

            align-items:
              flex-start;

            flex-direction:
              column;
          }


          .legend {

            align-self:
              flex-start;
          }


          .api-field {

            margin-left:
              0;

            width:
              100%;
          }


          .api-field input {

            flex:
              1;

            width:
              auto;
          }


          .run-btn {

            width:
              100%;
          }

        }

      `}</style>


      {/* ==================================================== */}
      {/* MAIN CONTAINER */}
      {/* ==================================================== */}

      <div className="tfbs-wrap">


        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <div className="tfbs-header">

          <div>

            <h1>
              TFBS prediction console
            </h1>


            <div className="sub">

              CNN-based transcription factor
              binding site prediction

            </div>

          </div>


          {/* DNA legend */}

          <div className="legend">

            <span>

              <span
                className="dot"
                style={{
                  background:
                    BASE_COLORS.A
                }}
              />

              A

            </span>


            <span>

              <span
                className="dot"
                style={{
                  background:
                    BASE_COLORS.T
                }}
              />

              T

            </span>


            <span>

              <span
                className="dot"
                style={{
                  background:
                    BASE_COLORS.C
                }}
              />

              C

            </span>


            <span>

              <span
                className="dot"
                style={{
                  background:
                    BASE_COLORS.G
                }}
              />

              G

            </span>

          </div>

        </div>


        {/* ================================================== */}
        {/* MAIN GRID */}
        {/* ================================================== */}

        <div className="tfbs-grid">


          {/* ================================================= */}
          {/* LEFT: SEQUENCE INPUT */}
          {/* ================================================= */}

          <div className="tfbs-panel">

            <h2>
              Sequence input
            </h2>


            {/* DNA input */}

            <textarea

              placeholder={
                "Enter a 101 bp DNA sequence — A, T, C, G only"
              }

              spellCheck={false}

              value={sequence}

              onChange={(event) => {

                setSequence(
                  event.target.value
                );

                setError("");

                setResult(null);

              }}

            />


            {/* ================================================= */}
            {/* SEQUENCE PREVIEW */}
            {/* ================================================= */}

            <div className="preview">

              <SequencePreview
                sequence={
                  sequence
                    .trim()
                    .toUpperCase()
                }
              />

            </div>


            {/* ================================================= */}
            {/* EXAMPLE BUTTONS */}
            {/* ================================================= */}

            <div className="examples">


              {/* TATA */}

              <button
                className="ex-btn"
                onClick={() =>
                  insertExample("tata")
                }
              >

                insert TATA box example

              </button>


              {/* GC */}

              <button
                className="ex-btn"
                onClick={() =>
                  insertExample("gc")
                }
              >

                insert GC box example

              </button>


              {/* CAAT */}

              <button
                className="ex-btn"
                onClick={() =>
                  insertExample("caat")
                }
              >

                insert CAAT box example

              </button>


              {/* Random */}

              <button
                className="ex-btn"
                onClick={() =>
                  insertExample("random")
                }
              >

                random 101 bp sequence

              </button>

            </div>


            {/* ================================================= */}
            {/* PREDICTION ROW */}
            {/* ================================================= */}

            <div className="tfbs-row">


              {/* Run prediction */}

              <button

                className="run-btn"

                onClick={
                  runPrediction
                }

                disabled={
                  loading
                }

              >

                {
                  loading
                    ? "running CNN..."
                    : "run CNN prediction"
                }

              </button>


              {/* API URL */}

              <div className="api-field">

                <span>
                  API
                </span>


                <input

                  value={
                    apiUrl
                  }

                  spellCheck={false}

                  onChange={(event) =>
                    setApiUrl(
                      event.target.value
                    )
                  }

                />

              </div>

            </div>


            {/* ================================================= */}
            {/* ERROR */}
            {/* ================================================= */}

            <div className="msg">

              {error}

            </div>


            {/* ================================================= */}
            {/* LOADING ANIMATION */}
            {/* ================================================= */}

            {
              loading && (

                <div className="scan-line" />

              )
            }

          </div>


          {/* ================================================= */}
          {/* RIGHT: RESULTS */}
          {/* ================================================= */}

          <div className="tfbs-panel">

            <h2>
              CNN prediction results
            </h2>


            {/* ================================================= */}
            {/* EMPTY STATE */}
            {/* ================================================= */}

            {
              !result &&
              !loading && (

                <div className="empty-state">

                  Run a 101 bp sequence to see
                  binding probability and
                  CNN saliency.

                </div>

              )
            }


            {/* ================================================= */}
            {/* RESULTS */}
            {/* ================================================= */}

            {
              result && (

                <div className="results-body">


                  {/* ========================================= */}
                  {/* PROBABILITY */}
                  {/* ========================================= */}

                  <div className="gauge-row">


                    <Gauge
                      probability={
                        result.binding_probability
                      }
                    />


                    <div>


                      {/* Sequence length */}

                      <div className="meta-line">

                        Input length:
                        {" "}
                        {result.input_length}
                        {" "}
                        bp

                      </div>


                      {/* Prediction */}

                      <div className="prediction-label">

                        {
                          result.prediction_label
                        }

                      </div>


                      {/* Model */}

                      <div className="model-badge">

                        Model:
                        {" "}
                        {result.model}

                      </div>

                    </div>

                  </div>


                  {/* ========================================= */}
                  {/* MOTIF RESULTS */}
                  {/* ========================================= */}

                  {
                    result.motifs_detected &&
                    result.motifs_detected.length > 0
                      ? (

                        <div className="motif-list">

                          {
                            result.motifs_detected.map(
                              (
                                motif,
                                index
                              ) => (

                                <span
                                  key={motif}
                                  className="motif-chip"
                                  style={{
                                    animationDelay:
                                      `${index * 80}ms`
                                  }}
                                >

                                  {motif}

                                </span>

                              )
                            )
                          }

                        </div>

                      )
                      : (

                        <div className="no-motifs">

                          No known motifs detected.

                        </div>

                      )
                  }


                  {/* ========================================= */}
                  {/* SALIENCY MAP */}
                  {/* ========================================= */}

                  <SaliencyTrack

                    saliency={
                      result
                        .visualizations
                        ?.saliency_map || []
                    }

                    sequence={
                      result.sequence
                    }

                  />


                </div>

              )
            }

          </div>

        </div>

      </div>

    </div>
  );
}