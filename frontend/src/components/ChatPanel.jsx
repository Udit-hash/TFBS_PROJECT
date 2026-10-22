import { useEffect, useRef, useState } from 'react';
import { postJSON } from '../lib/api';
import { C, mono, chip } from './ui';

const GENERAL_STARTERS = [
  'What is DNA, in simple words?',
  'What is a transcription factor?',
  'How does this model make a prediction?',
  'Why does G–C have 3 hydrogen bonds but A–T only 2?',
];

const RESULT_STARTERS = [
  'Why did my sequence get this score?',
  'What do the bright base pairs in the 3D view mean?',
  'Explain the mutation (ISM) results for my sequence',
  'Is the 3D view a real simulation?',
];

/** Renders **bold** and line breaks safely (no HTML injection). */
function RichText({ text }) {
  return text.split('\n').map((line, li) => (
    <p key={li} style={{ margin: line ? '0 0 6px' : '0 0 2px' }}>
      {line.split(/(\*\*[^*]+\*\*)/g).map((part, pi) =>
        part.startsWith('**') && part.endsWith('**')
          ? <strong key={pi} style={{ color: C.text }}>{part.slice(2, -2)}</strong>
          : <span key={pi}>{part}</span>
      )}
    </p>
  ));
}

export default function ChatPanel({ open, onOpenChange, prediction, pendingQuestion, onPendingHandled }) {
  const [mode, setMode] = useState('beginner');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, busy]);

  const send = async (question) => {
    const q = question.trim();
    if (!q || busy) return;
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: 'user', content: q }]);
    setInput('');
    setBusy(true);
    try {
      const data = await postJSON('/chat', { question: q, mode, history, prediction });
      setMessages((m) => [...m, { role: 'assistant', content: data.answer, source: data.source, refs: data.references, notice: data.notice }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', content: `Sorry, I couldn't reach the backend (${e.message}). Is it running?`, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  // Questions sent from other parts of the page ("Ask the tutor about this")
  useEffect(() => {
    if (pendingQuestion && open) {
      send(pendingQuestion);
      onPendingHandled?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingQuestion, open]);

  const starters = prediction ? RESULT_STARTERS : GENERAL_STARTERS;

  if (!open) {
    return (
      <button
        onClick={() => onOpenChange(true)}
        style={{
          position: 'fixed', right: '24px', bottom: '24px', zIndex: 40,
          background: C.accentStrong, color: '#fff', border: 'none', borderRadius: '999px',
          padding: '12px 20px', fontSize: '14px', fontWeight: 600, cursor: 'pointer',
          boxShadow: '0 10px 30px rgba(2,132,199,0.35)',
        }}
      >
        💬 Ask the DNA tutor
      </button>
    );
  }

  return (
    <aside
      aria-label="DNA tutor chat"
      style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, zIndex: 50,
        width: 'min(420px, 100vw)', background: '#0e1526', borderLeft: `1px solid ${C.border}`,
        display: 'flex', flexDirection: 'column', boxShadow: '-20px 0 40px rgba(0,0,0,0.35)',
      }}
    >
      <header style={{ padding: '16px 18px', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 700, color: C.text }}>DNA tutor</div>
            <div style={{ fontSize: '12px', color: C.muted, marginTop: '2px' }}>
              {prediction ? 'Knows your current prediction' : 'Run a prediction to ask about your sequence'}
            </div>
          </div>
          <button onClick={() => onOpenChange(false)} aria-label="Close chat" style={{ background: 'none', border: 'none', color: C.muted, fontSize: '20px', cursor: 'pointer' }}>×</button>
        </div>
        <div style={{ display: 'flex', gap: '6px', marginTop: '12px' }}>
          <button style={chip(mode === 'beginner')} onClick={() => setMode('beginner')}>Beginner</button>
          <button style={chip(mode === 'advanced')} onClick={() => setMode('advanced')}>Advanced</button>
        </div>
      </header>

      <div ref={listRef} style={{ flex: 1, overflowY: 'auto', padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {messages.length === 0 && (
          <div style={{ color: C.body, fontSize: '14px', lineHeight: 1.6 }}>
            <p style={{ marginBottom: '12px' }}>
              Hi! I can explain DNA, transcription factors and how this model reached its answer. No biology background needed.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {starters.map((s) => (
                <button key={s} onClick={() => send(s)} style={{ ...chip(false), textAlign: 'left', whiteSpace: 'normal', borderRadius: '10px', padding: '9px 12px', fontSize: '13px', color: C.body }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '88%' }}>
            <div style={{
              background: m.role === 'user' ? C.accentStrong : C.panel,
              color: m.role === 'user' ? '#fff' : m.error ? C.danger : C.body,
              border: m.role === 'user' ? 'none' : `1px solid ${C.border}`,
              borderRadius: m.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
              padding: '10px 13px', fontSize: '14px', lineHeight: 1.55,
            }}>
              {m.role === 'user' ? m.content : <RichText text={m.content} />}
            </div>
            {m.role === 'assistant' && !m.error && (
              <div style={{ fontSize: '10.5px', color: C.faint, fontFamily: mono, marginTop: '4px' }}>
                {m.source === 'offline' ? 'offline notes' : 'AI answer'}
                {m.refs?.length ? ` · based on: ${m.refs.slice(0, 2).join(', ')}` : ''}
                {m.notice ? ` · ${m.notice}` : ''}
              </div>
            )}
          </div>
        ))}
        {busy && <div style={{ color: C.muted, fontSize: '13px', fontFamily: mono }}>thinking…</div>}
      </div>

      <form
        onSubmit={(e) => { e.preventDefault(); send(input); }}
        style={{ padding: '12px 14px', borderTop: `1px solid ${C.border}`, display: 'flex', gap: '8px' }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input); } }}
          rows={2}
          maxLength={1000}
          placeholder="Ask anything about DNA binding…"
          style={{ flex: 1, resize: 'none', background: C.panelDeep, color: C.text, border: `1px solid ${C.borderStrong}`, borderRadius: '10px', padding: '10px', fontSize: '14px', fontFamily: 'inherit', outline: 'none' }}
        />
        <button type="submit" disabled={busy || !input.trim()} style={{ background: input.trim() ? C.accentStrong : C.borderStrong, color: '#fff', border: 'none', borderRadius: '10px', padding: '0 16px', fontWeight: 600, cursor: input.trim() ? 'pointer' : 'not-allowed' }}>
          Send
        </button>
      </form>
    </aside>
  );
}
