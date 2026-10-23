"""
TFBS tutor chatbot.

How it answers a question:
  1. Retrieve the most relevant entries from knowledge_base.json
     (simple keyword scoring, no heavy libraries).
  2. Summarise the user's current prediction result (if any) so the
     answer can talk about *their* sequence.
  3. If an LLM API key is configured (see .env.example), send the
     retrieved notes + prediction summary to an OpenAI-compatible chat
     API (Groq, Gemini, OpenAI, ...). Otherwise answer offline by
     composing the retrieved notes directly.

Only Python's standard library is used, so nothing new to install and
nothing runs in the background: each question is one short HTTPS request.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_PATH = HERE / "knowledge_base.json"
ENV_PATH = HERE.parent / ".env"

MAX_HISTORY_TURNS = 8
LLM_TIMEOUT_SECONDS = 30


# ============================================================
# CONFIG (.env loader without extra dependencies)
# ============================================================

def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(ENV_PATH)


def llm_settings() -> dict | None:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        return None
    return {
        "api_key": api_key,
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
    }


# ============================================================
# KNOWLEDGE BASE RETRIEVAL
# ============================================================

with open(KB_PATH, encoding="utf-8") as fh:
    KNOWLEDGE_BASE: list[dict] = json.load(fh)

_STOPWORDS = set(
    "a an the is are was were be been of to in on for and or what why how does do did "
    "it this that these those my me i you your with by at as from can could would should "
    "explain tell about please mean means meaning there here which who whom".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9\-]+", text.lower()) if t not in _STOPWORDS]


def retrieve_scored(question: str, k: int = 3) -> list[tuple[float, dict]]:
    q = question.lower()
    q_tokens = set(_tokens(question))
    scored = []
    for entry in KNOWLEDGE_BASE:
        score = 0.0
        for kw in entry["keywords"]:
            kw_l = kw.lower()
            if " " in kw_l or "-" in kw_l:
                if kw_l in q:
                    score += 4.0
            elif kw_l in q_tokens:
                score += 3.0
        title_tokens = set(_tokens(entry["title"]))
        score += 1.5 * len(q_tokens & title_tokens)
        body_tokens = set(_tokens(entry["beginner"] + " " + entry["advanced"]))
        score += 0.25 * len(q_tokens & body_tokens)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:k]


def retrieve(question: str, k: int = 3) -> list[dict]:
    return [e for _, e in retrieve_scored(question, k)]


_POSITION_RE = re.compile(r"\b(?:position|pos|base|bp)\s*#?\s*(\d{1,3})\b", re.I)


def focus_position(question: str, ctx: dict | None) -> dict | None:
    """If the user asks about a specific position, collect everything we know about it."""
    if not ctx:
        return None
    m = _POSITION_RE.search(question)
    if not m:
        return None
    pos = int(m.group(1))
    seq = ctx.get("sequence") or ""
    sal = ctx.get("saliency") or []
    eff = ctx.get("ism_position_effect") or []
    if pos >= len(seq):
        return None
    info = {"position": pos, "base": seq[pos], "complement": {"A": "T", "T": "A", "C": "G", "G": "C"}.get(seq[pos], "?")}
    if pos < len(sal):
        info["saliency"] = sal[pos]
        info["rank"] = sum(1 for v in sal if v >= sal[pos])  # ties count against, so flat regions rank low
        info["length"] = len(sal)
    if pos < len(eff):
        info["worst_delta"] = eff[pos]
    start = ctx.get("window_start")
    core = ctx.get("core_sequence") or ""
    info["in_core"] = start is not None and start <= pos < start + len(core)
    info["mutations"] = [mu for mu in (ctx.get("top_mutations") or []) if mu.get("position") == pos]
    return info


def describe_focus(f: dict) -> str:
    parts = [f"Position {f['position']} is {f['base']} (paired with {f['complement']})"]
    if "saliency" in f:
        parts.append(f"saliency {f['saliency']:.2f}, rank {f['rank']} of {f['length']}")
    if "worst_delta" in f:
        parts.append(f"most damaging single mutation changes p(bind) by {f['worst_delta']:+.3f}")
    parts.append("inside the peak window" if f["in_core"] else "outside the peak window")
    return "; ".join(parts) + "."


# ============================================================
# PREDICTION CONTEXT
# ============================================================

def summarise_prediction(ctx: dict | None) -> str:
    """Turn the compact result object sent by the frontend into plain text."""
    if not ctx:
        return ""
    lines = []
    prob = ctx.get("binding_probability")
    if prob is not None:
        lines.append(f"Binding probability: {prob * 100:.1f}% ({ctx.get('prediction_label', '')})")
    if ctx.get("confidence_interval"):
        lo, hi = ctx["confidence_interval"]
        lines.append(f"95% interval (MC dropout): {lo * 100:.1f}-{hi * 100:.1f}%; decision cut-off {ctx.get('threshold', 0.5):.2f}")
    if ctx.get("known_hits"):
        lines.append("JASPAR motifs found: " + ", ".join(
            f"{h['name']} at {h['start']}-{h['end'] - 1} ({h['strand']}, match {h['score']:.2f}, importance {h['saliency']:.2f})"
            for h in ctx["known_hits"][:4]))
    if ctx.get("core_sequence"):
        lines.append(
            f"Most important window: {ctx['core_sequence']} starting at position "
            f"{ctx.get('window_start')} (closest known profile: {ctx.get('matched_jaspar_id', 'unknown')})"
        )
    if ctx.get("motifs_detected"):
        lines.append("Exact known motifs found in the sequence: " + ", ".join(ctx["motifs_detected"]))
    top = ctx.get("top_positions") or []
    if top:
        lines.append(
            "Highest-saliency positions: "
            + ", ".join(f"{p['position']}{p['base']} ({p['score']:.2f})" for p in top[:6])
        )
    muts = ctx.get("top_mutations") or []
    if muts:
        lines.append(
            "Largest in-silico-mutation effects: "
            + ", ".join(f"pos {m['position']} {m['mutation']} Δp={m['delta_p']:+.3f}" for m in muts[:5])
        )
    return "\n".join(lines)


_RESULT_WORDS = re.compile(
    r"\b(my|this|result|sequence|score|prediction|position|pos|why|highlight|mutation|predicted|probability)\b",
    re.I,
)


def _explain_focus_offline(f: dict, mode: str) -> str:
    hb = 2 if f["base"] in "AT" else 3
    if mode != "beginner":
        return describe_focus(f) + (
            " High saliency plus a large negative Δp means the model depends on this base. "
            "Agreement between the gradient and the mutation test makes the attribution more trustworthy."
            if f.get("worst_delta", 0) < -0.05 else
            " The mutation effect is small, so the model does not depend strongly on this base on its own."
        )
    rank = f.get("rank")
    lines = [f"Position {f['position']} is the letter {f['base']}, paired with {f['complement']} on the other strand by {hb} hydrogen bonds."]
    if rank is not None:
        if rank <= 5:
            lines.append(f"It is one of the most important letters for the model: #{rank} out of {f['length']}.")
        elif rank <= 20:
            lines.append(f"The model finds it fairly important (#{rank} out of {f['length']}).")
        else:
            lines.append(f"The model doesn't rely on it much (#{rank} out of {f['length']}).")
    if f["in_core"]:
        lines.append("It sits inside the stretch of DNA the model thinks the protein recognises, the 'peak window'.")
    d = f.get("worst_delta")
    if d is not None and d < -0.05:
        lines.append(f"If we swap this letter for the worst alternative, the binding score drops by about {abs(d) * 100:.0f} points. "
                     "It's like changing one letter in a password: the protein no longer recognises the word as well.")
    elif d is not None:
        lines.append("Changing this letter barely moves the score, so on its own it isn't critical.")
    return " ".join(lines)


def _explain_result_offline(ctx: dict, mode: str) -> str:
    prob = ctx.get("binding_probability", 0)
    pct = f"{prob * 100:.1f}%"
    core = ctx.get("core_sequence")
    start = ctx.get("window_start")
    match = ctx.get("matched_jaspar_id", "")
    muts = ctx.get("top_mutations") or []

    if mode == "beginner":
        verdict = (
            "the model thinks a transcription factor is likely to stick here"
            if prob >= 0.5
            else "the model thinks a transcription factor is unlikely to stick here"
        )
        parts = [f"Your sequence scored {pct}, so {verdict}."]
        if core:
            parts.append(
                f"The letters it paid most attention to are {core} (starting at position {start}). "
                "In the 3D view, those are the brightest base pairs."
            )
        if match and "De Novo" not in match:
            parts.append(f"That stretch looks like a known binding word: {match}.")
        if muts:
            m = muts[0]
            direction = "drop" if m["delta_p"] < 0 else "rise"
            parts.append(
                f"When we change position {m['position']} ({m['mutation']}), the score would "
                f"{direction} by {abs(m['delta_p']) * 100:.1f} points. That tells us this letter really matters."
            )
        return " ".join(parts)

    parts = [f"p(bind) = {prob:.3f} ({ctx.get('prediction_label', '')})."]
    if core:
        parts.append(f"Peak ShiftSmooth window: {core} @ {start}; profile lookup: {match}.")
    if muts:
        parts.append(
            "Top ISM effects: "
            + "; ".join(f"{m['position']} {m['mutation']} Δp={m['delta_p']:+.3f}" for m in muts[:3])
            + "."
        )
    parts.append("Large negative Δp at the saliency peak means the attribution and ISM agree about the core.")
    return " ".join(parts)


# ============================================================
# ANSWERING
# ============================================================

SYSTEM_PROMPT = """You are the built-in tutor of a TFBS (transcription factor binding site) prediction web app built by an undergraduate team.
The app's model (by default the team's CNN + BiLSTM + Transformer v2, test AUROC 0.934; the user can switch to CNN + BiLSTM, 0.939, or a CNN, 0.924) reads a 101 bp DNA sequence and predicts the probability that the transcription factor FOSL2 (AP-1 family) binds it; it was trained on ENCODE ChIP-seq data from HepG2 liver cells. It explains predictions with ShiftSmooth saliency, in-silico mutagenesis (ISM) and a 3D DNA view where base pairs are coloured by importance.

Rules:
- Mode is "{mode}". In beginner mode: assume no biology or AI background, use everyday analogies, define any technical word the first time you use it, and keep it under 150 words. In advanced mode: be precise and technical, still under 220 words.
- When the user asks about their result, use ONLY the numbers in "Current prediction". Never invent positions, scores or motifs.
- The model only reads DNA letters. It does not simulate atoms or physical bonds. The hydrogen bonds in the 3D view are the standard A–T (2) and G–C (3) pairs, not computed contacts. Say so if the user asks whether the 3D view is "real".
- Base your explanations on the reference notes when they are relevant. If you are unsure, say so.
- This is a research and learning tool, not medical advice.
- Stay on topic: DNA, gene regulation, transcription factors, this model and its explanations. Politely steer other questions back.

Reference notes:
{notes}

Current prediction:
{prediction}
"""


def _call_llm(settings: dict, system: str, history: list[dict], question: str) -> str:
    messages = [{"role": "system", "content": system}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])[:2000]})
    messages.append({"role": "user", "content": question})

    body = json.dumps(
        {"model": settings["model"], "messages": messages, "temperature": 0.3, "max_tokens": 450}
    ).encode("utf-8")
    req = urllib.request.Request(
        settings["base_url"] + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _offline_answer(question: str, scored: list[tuple[float, dict]], ctx: dict | None, mode: str,
                    focus: dict | None) -> str:
    key = "beginner" if mode == "beginner" else "advanced"
    pieces = []
    if focus:
        pieces.append(_explain_focus_offline(focus, mode))
    elif ctx and _RESULT_WORDS.search(question):
        pieces.append(_explain_result_offline(ctx, mode))
    if not focus:
        top = scored[0][0] if scored else 0
        relevant = [e for sc, e in scored if sc >= 0.6 * top]
        for entry in relevant[: (1 if pieces else 2)]:
            pieces.append(f"**{entry['title']}**: {entry[key]}")
    if not pieces:
        pieces.append(
            "I'm not sure about that one. Try asking about DNA basics, transcription factors, motifs, "
            "the binding score, the saliency map, mutations (ISM) or the 3D view."
        )
    return "\n\n".join(pieces)


def answer(question: str, mode: str = "beginner", history: list[dict] | None = None,
           prediction: dict | None = None) -> dict:
    question = (question or "").strip()[:1000]
    mode = "advanced" if mode == "advanced" else "beginner"
    history = history or []

    scored = retrieve_scored(question, k=3)
    notes = [e for _, e in scored]
    focus = focus_position(question, prediction)
    notes_text = "\n".join(f"- {e['title']}: {e['advanced']}" for e in notes) or "(none matched)"
    prediction_text = summarise_prediction(prediction) or "(the user has not run a prediction yet)"
    if focus:
        prediction_text += "\nAsked-about position: " + describe_focus(focus)

    settings = llm_settings()
    source = "offline"
    reply = None
    error = None
    if settings:
        try:
            system = SYSTEM_PROMPT.format(mode=mode, notes=notes_text, prediction=prediction_text)
            reply = _call_llm(settings, system, history, question)
            source = "llm"
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            error = f"LLM unavailable ({exc.__class__.__name__}); answered from the built-in notes."

    if reply is None:
        reply = _offline_answer(question, scored, prediction, mode, focus)

    return {
        "answer": reply,
        "mode": mode,
        "source": source,
        "references": [] if (focus and source == "offline") else [e["title"] for e in notes],
        "notice": error,
    }
