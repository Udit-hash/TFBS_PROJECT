// Requests go through the Vite dev-server proxy (/api -> http://127.0.0.1:8001, see vite.config.js).
// Override with VITE_API_BASE in frontend/.env if the backend lives elsewhere.
export const API_BASE = import.meta.env?.VITE_API_BASE || '/api';

export async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const err = await res.json();
      detail = typeof err.detail === 'string' ? err.detail : detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

/** Compact summary of a /predict result that the chatbot can reason about. */
export function summarisePrediction(result, sequence) {
  if (!result) return null;
  const viz = result.visualizations || {};
  const saliency = viz.saliency_map || [];
  const top = saliency
    .map((score, position) => ({ position, base: sequence?.[position] || '?', score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
  return {
    binding_probability: result.binding_probability,
    prediction_label: result.prediction_label,
    motifs_detected: result.motifs_detected || [],
    core_sequence: viz.sequence_logo?.core_sequence,
    window_start: viz.sequence_logo?.window_start,
    matched_jaspar_id: viz.sequence_logo?.matched_jaspar_id,
    top_positions: top,
    top_mutations: (viz.in_silico_mutagenesis || []).slice(0, 5),
    // Full per-position data so the tutor can answer "why is position N important?"
    sequence,
    saliency: saliency.map((v) => Math.round(v * 1000) / 1000),
    ism_position_effect: viz.ism_position_effect || [],
  };
}
