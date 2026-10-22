/**
 * Lightweight 3D DNA double-helix renderer (plain Canvas 2D, no WebGL, no libraries).
 *
 * - Idealised B-DNA geometry: 10.5 bp per turn, 3.38 Å rise, ~9 Å radius,
 *   with the two backbones offset so a wide major groove and a narrow minor groove appear.
 * - Each base pair is drawn as two coloured half-rungs (base + complement) joined
 *   by its Watson–Crick hydrogen bonds: 2 for A–T, 3 for G–C.
 * - Colours can show nucleotide identity, ShiftSmooth saliency, or ISM impact.
 * - Draws ONLY when something changes (drag, click, resize, data change).
 *   No animation loop runs in the background, so laptops stay cool.
 */

const RISE = 3.38;               // Å per base pair
const TWIST = (2 * Math.PI) / 10.5;
const R_BACKBONE = 9.4;          // phosphate radius (Å)
const R_BASE = 8.0;              // where the base attaches
const STRAND_OFFSET = 2.35;      // radians between the two strands (creates the minor groove)
const CAMERA_DIST = 260;

export const BASE_COLORS = { A: '#10b981', C: '#3b82f6', G: '#f59e0b', T: '#ef4444' };
export const COMPLEMENT = { A: 'T', T: 'A', C: 'G', G: 'C' };
export const HBONDS = { A: 2, T: 2, C: 3, G: 3 };

const HEAT_STOPS = [
  [0.0, [51, 65, 85]],    // slate: not important
  [0.35, [14, 165, 233]], // sky
  [0.7, [250, 204, 21]],  // yellow
  [1.0, [244, 63, 94]],   // rose: most important
];

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function heatColor(v) {
  const x = Math.max(0, Math.min(1, v || 0));
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const [p1, c1] = HEAT_STOPS[i];
    const [p0, c0] = HEAT_STOPS[i - 1];
    if (x <= p1) {
      const t = (x - p0) / (p1 - p0);
      return c0.map((c, k) => Math.round(c + (c1[k] - c) * t));
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}

export function rgbToCss(rgb) {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

export class HelixRenderer {
  constructor(canvas, { onSelect, background = '#0b1220' } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSelect = onSelect;
    this.bg = hexToRgb(background);

    this.sequence = '';
    this.saliency = [];
    this.ismEffect = [];
    this.coreStart = null;
    this.coreLen = 0;
    this.windowStart = 0;
    this.windowLen = 24;
    this.colorMode = 'importance';
    this.selected = null;

    this.spin = 0.35;
    this.yaw = -0.35;
    this.zoomFactor = 1;
    this.hitPoints = [];
    this._frame = null;
    this._dragging = null;

    this._onDown = this._onDown.bind(this);
    this._onMove = this._onMove.bind(this);
    this._onUp = this._onUp.bind(this);
    canvas.addEventListener('pointerdown', this._onDown);
    window.addEventListener('pointermove', this._onMove);
    window.addEventListener('pointerup', this._onUp);

    this._resizeObserver = new ResizeObserver(() => this.requestRender());
    this._resizeObserver.observe(canvas);
  }

  destroy() {
    this.canvas.removeEventListener('pointerdown', this._onDown);
    window.removeEventListener('pointermove', this._onMove);
    window.removeEventListener('pointerup', this._onUp);
    this._resizeObserver.disconnect();
    if (this._frame) cancelAnimationFrame(this._frame);
  }

  // ---------------------------------------------------------------- state

  setData({ sequence, saliency, ismEffect, coreStart, coreLen }) {
    this.sequence = sequence || '';
    this.saliency = saliency || [];
    this.ismEffect = ismEffect || [];
    this.coreStart = Number.isInteger(coreStart) ? coreStart : null;
    this.coreLen = coreLen || 0;
    const neg = this.ismEffect.filter((d) => d < 0).map((d) => -d);
    this._ismMax = neg.length ? Math.max(...neg) : 0;
    this.requestRender();
  }

  setWindow(start, len) {
    const n = this.sequence.length;
    this.windowLen = Math.max(6, Math.min(len, n || len));
    this.windowStart = Math.max(0, Math.min(start, Math.max(0, n - this.windowLen)));
    this.requestRender();
  }

  setColorMode(mode) { this.colorMode = mode; this.requestRender(); }
  setSelected(pos) { this.selected = pos; this.requestRender(); }
  setZoom(z) { this.zoomFactor = Math.max(0.6, Math.min(3, z)); this.requestRender(); }
  rotateBy(dSpin, dYaw = 0) {
    this.spin += dSpin;
    this.yaw = Math.max(-1.25, Math.min(1.25, this.yaw + dYaw));
    this.requestRender();
  }
  resetView() { this.spin = 0.35; this.yaw = -0.35; this.zoomFactor = 1; this.requestRender(); }

  importanceAt(i) {
    if (this.colorMode === 'ism') {
      const d = this.ismEffect[i] || 0;
      return this._ismMax > 0 && d < 0 ? -d / this._ismMax : 0;
    }
    return this.saliency[i] || 0;
  }

  // ---------------------------------------------------------- interaction

  _onDown(e) {
    this._dragging = { x: e.clientX, y: e.clientY, moved: 0 };
    this.canvas.setPointerCapture?.(e.pointerId);
  }

  _onMove(e) {
    if (!this._dragging) return;
    const dx = e.clientX - this._dragging.x;
    const dy = e.clientY - this._dragging.y;
    this._dragging.x = e.clientX;
    this._dragging.y = e.clientY;
    this._dragging.moved += Math.abs(dx) + Math.abs(dy);
    // vertical drag spins the helix around its own axis, horizontal drag tilts it
    this.rotateBy(dy * 0.012, dx * 0.006);
  }

  _onUp(e) {
    if (!this._dragging) return;
    const wasClick = this._dragging.moved < 5;
    this._dragging = null;
    if (wasClick && e.target === this.canvas) this._pick(e);
  }

  _pick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    let best = null;
    for (const h of this.hitPoints) {
      const d = Math.hypot(h.x - x, h.y - y);
      if (d < 20 && (!best || d - h.z * 0.05 < best.score)) best = { score: d - h.z * 0.05, pos: h.pos };
    }
    const pos = best ? best.pos : null;
    this.selected = pos;
    this.requestRender();
    this.onSelect?.(pos);
  }

  // -------------------------------------------------------------- drawing

  requestRender() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = null;
      this.render();
    });
  }

  render() {
    const { canvas, ctx } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    if (!W || !H) return;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    this.hitPoints = [];
    if (!this.sequence) return;

    const start = this.windowStart;
    const len = Math.min(this.windowLen, this.sequence.length - start);
    const center = start + (len - 1) / 2;

    const baseZoom = Math.min((0.86 * W) / (len * RISE), (0.62 * H) / (2 * R_BACKBONE));
    const zoom = baseZoom * this.zoomFactor;
    const cx = W / 2;
    const cy = H * 0.46;
    const cosY = Math.cos(this.yaw);
    const sinY = Math.sin(this.yaw);

    // 3D point along the helix (fractional index allowed for smooth backbones)
    const helixPoint = (idx, radius, angleOffset) => {
      const theta = idx * TWIST + this.spin + angleOffset;
      return [(idx - center) * RISE, radius * Math.cos(theta), radius * Math.sin(theta)];
    };
    const project = ([x, y, z]) => {
      const xr = x * cosY + z * sinY;
      const zr = -x * sinY + z * cosY;
      const f = CAMERA_DIST / (CAMERA_DIST - zr);
      return { x: cx + xr * f * zoom, y: cy - y * f * zoom, z: zr, s: f * zoom };
    };
    const depthShade = (z) => 0.38 + 0.62 * Math.max(0, Math.min(1, (z + R_BACKBONE) / (2 * R_BACKBONE)));
    const shade = (rgb, b, alpha = 1) => {
      const c = rgb.map((v, k) => Math.round(this.bg[k] + (v - this.bg[k]) * b));
      return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
    };

    const prims = [];
    const strandRgb = [hexToRgb('#cbd5e1'), hexToRgb('#94a3b8')];

    // ---- backbones (smooth, sub-sampled)
    const STEPS = 5;
    [0, STRAND_OFFSET].forEach((off, s) => {
      for (let i = start - 0.5; i < start + len - 0.5; i += 1 / STEPS) {
        const a = project(helixPoint(i, R_BACKBONE, off));
        const b = project(helixPoint(i + 1 / STEPS, R_BACKBONE, off));
        prims.push({ type: 'seg', a, b, z: (a.z + b.z) / 2, w: 1.35, rgb: strandRgb[s] });
      }
    });

    // ---- base pairs
    for (let i = start; i < start + len; i++) {
      const base = this.sequence[i];
      const comp = COMPLEMENT[base] || 'N';
      const p1 = helixPoint(i, R_BACKBONE, 0);
      const p2 = helixPoint(i, R_BACKBONE, STRAND_OFFSET);
      const e1 = helixPoint(i, R_BASE, 0);
      const e2 = helixPoint(i, R_BASE, STRAND_OFFSET);
      const mid = e1.map((v, k) => (v + e2[k]) / 2);
      const m1 = e1.map((v, k) => v + (mid[k] - v) * 0.7);
      const m2 = e2.map((v, k) => v + (mid[k] - v) * 0.7);

      const imp = this.importanceAt(i);
      const heat = heatColor(imp);
      const c1 = this.colorMode === 'bases' ? hexToRgb(BASE_COLORS[base] || '#64748b') : heat;
      const c2 = this.colorMode === 'bases' ? hexToRgb(BASE_COLORS[comp] || '#64748b') : heat;
      const isSel = this.selected === i;
      const inCore = this.coreStart !== null && i >= this.coreStart && i < this.coreStart + this.coreLen;

      const P1 = project(p1), P2 = project(p2);
      const E1 = project(e1), E2 = project(e2);
      const M1 = project(m1), M2 = project(m2);

      prims.push({ type: 'seg', a: P1, b: E1, z: (P1.z + E1.z) / 2, w: 1.2, rgb: strandRgb[0] });
      prims.push({ type: 'seg', a: P2, b: E2, z: (P2.z + E2.z) / 2, w: 1.2, rgb: strandRgb[1] });
      prims.push({ type: 'rung', a: E1, b: M1, z: (E1.z + M1.z) / 2, w: 1.7, rgb: c1, sel: isSel, glow: imp, letter: base, core: inCore });
      prims.push({ type: 'rung', a: E2, b: M2, z: (E2.z + M2.z) / 2, w: 1.7, rgb: c2, sel: isSel, glow: imp, letter: comp, core: inCore });

      // hydrogen bonds bridging the gap between the two bases
      const n = HBONDS[base] || 0;
      const offsets = n === 3 ? [-0.6, 0, 0.6] : [-0.33, 0.33];
      for (let k = 0; k < n; k++) {
        const o = offsets[k];
        const a = project([m1[0] + o, m1[1], m1[2]]);
        const b = project([m2[0] + o, m2[1], m2[2]]);
        prims.push({ type: 'hbond', a, b, z: (a.z + b.z) / 2 + 0.01 });
      }

      prims.push({ type: 'ball', p: P1, z: P1.z, r: 1.0, rgb: strandRgb[0] });
      prims.push({ type: 'ball', p: P2, z: P2.z, r: 1.0, rgb: strandRgb[1] });

      const MID = project(mid);
      this.hitPoints.push({ x: MID.x, y: MID.y, z: MID.z, pos: i });
    }

    prims.sort((p, q) => p.z - q.z);
    ctx.lineCap = 'round';

    for (const p of prims) {
      const b = depthShade(p.z);
      if (p.type === 'seg') {
        ctx.strokeStyle = shade(p.rgb, b * 0.85);
        ctx.lineWidth = Math.max(1, p.w * p.a.s);
        ctx.beginPath(); ctx.moveTo(p.a.x, p.a.y); ctx.lineTo(p.b.x, p.b.y); ctx.stroke();
      } else if (p.type === 'ball') {
        const r = Math.max(1.5, p.r * p.p.s);
        const g = ctx.createRadialGradient(p.p.x - r * 0.35, p.p.y - r * 0.35, r * 0.1, p.p.x, p.p.y, r);
        g.addColorStop(0, shade([255, 255, 255], b * 0.9));
        g.addColorStop(1, shade(p.rgb, b * 0.75));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(p.p.x, p.p.y, r, 0, Math.PI * 2); ctx.fill();
      } else if (p.type === 'rung') {
        const w = Math.max(2, p.w * p.a.s);
        if (p.sel) {
          ctx.strokeStyle = 'rgba(255,255,255,0.95)';
          ctx.lineWidth = w + 4;
          ctx.beginPath(); ctx.moveTo(p.a.x, p.a.y); ctx.lineTo(p.b.x, p.b.y); ctx.stroke();
        } else if (this.colorMode !== 'bases' && p.glow > 0.6) {
          ctx.strokeStyle = shade(p.rgb, b, 0.25 * p.glow);
          ctx.lineWidth = w + 6 * p.glow;
          ctx.beginPath(); ctx.moveTo(p.a.x, p.a.y); ctx.lineTo(p.b.x, p.b.y); ctx.stroke();
        }
        ctx.strokeStyle = shade(p.rgb, b);
        ctx.lineWidth = w;
        ctx.beginPath(); ctx.moveTo(p.a.x, p.a.y); ctx.lineTo(p.b.x, p.b.y); ctx.stroke();

        const fontPx = Math.min(13, Math.max(0, 1.7 * p.a.s));
        if (fontPx >= 8 && b > 0.62) {
          ctx.fillStyle = shade([241, 245, 249], b);
          ctx.font = `600 ${fontPx}px "IBM Plex Mono", ui-monospace, monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const lx = p.a.x + (p.b.x - p.a.x) * 0.45;
          const ly = p.a.y + (p.b.y - p.a.y) * 0.45;
          ctx.fillText(p.letter, lx, ly - w * 0.9);
        }
      } else if (p.type === 'hbond') {
        ctx.strokeStyle = shade([226, 232, 240], b, 0.85);
        ctx.lineWidth = Math.max(1, 0.32 * p.a.s);
        ctx.setLineDash([Math.max(1.5, 0.35 * p.a.s), Math.max(1.5, 0.3 * p.a.s)]);
        ctx.beginPath(); ctx.moveTo(p.a.x, p.a.y); ctx.lineTo(p.b.x, p.b.y); ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // ---- overlays: position ruler and peak window bracket (not depth-sorted)
    const rulerY = (idx) => project([(idx - center) * RISE, -R_BACKBONE - 3.2, 0]);
    ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = start; i < start + len; i++) {
      if (i % 5 !== 0 && i !== start && i !== start + len - 1) continue;
      const q = rulerY(i);
      ctx.fillStyle = 'rgba(148,163,184,0.85)';
      ctx.fillText(String(i), q.x, q.y + 4);
      ctx.fillRect(q.x - 0.5, q.y - 2, 1, 5);
    }
    if (this.coreStart !== null && this.coreLen) {
      const s = Math.max(this.coreStart, start);
      const e = Math.min(this.coreStart + this.coreLen - 1, start + len - 1);
      if (s <= e) {
        const a = rulerY(s - 0.4), b = rulerY(e + 0.4);
        ctx.strokeStyle = 'rgba(244,63,94,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y - 6); ctx.lineTo(a.x, a.y - 1); ctx.lineTo(b.x, b.y - 1); ctx.lineTo(b.x, b.y - 6);
        ctx.stroke();
        ctx.fillStyle = 'rgba(251,113,133,0.95)';
        ctx.fillText('peak window', (a.x + b.x) / 2, a.y + 18);
      }
    }
  }
}
