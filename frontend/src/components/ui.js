// Shared style tokens for the dashboard (matches the existing dark theme).
export const C = {
  page: '#0b0f19',
  panel: '#131b2e',
  panelDeep: '#090d16',
  canvas: '#0b1220',
  border: '#1e293b',
  borderStrong: '#334155',
  text: '#f8fafc',
  body: '#e2e8f0',
  muted: '#94a3b8',
  faint: '#64748b',
  accent: '#7dd3fc',
  accentStrong: '#0284c7',
  danger: '#f87171',
  good: '#34d399',
};

export const mono = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace';

// Matches the .panel style in App.css
export const panelStyle = {
  background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.8))',
  border: '1px solid rgba(148, 163, 184, 0.18)',
  borderRadius: '18px',
  boxShadow: '0 18px 40px rgba(2, 6, 23, 0.35)',
  padding: '24px',
};

export const sectionTitle = {
  fontSize: '15px',
  fontWeight: 600,
  color: C.text,
  fontFamily: mono,
};

export function chip(active) {
  return {
    background: active ? 'rgba(56,189,248,0.15)' : 'transparent',
    color: active ? C.accent : C.muted,
    border: `1px solid ${active ? 'rgba(56,189,248,0.45)' : C.borderStrong}`,
    borderRadius: '999px',
    padding: '5px 12px',
    fontSize: '12px',
    fontFamily: mono,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  };
}

export const iconButton = {
  background: C.panelDeep,
  color: C.body,
  border: `1px solid ${C.borderStrong}`,
  borderRadius: '8px',
  minWidth: '32px',
  height: '32px',
  padding: '0 8px',
  fontSize: '13px',
  fontFamily: mono,
  cursor: 'pointer',
};
