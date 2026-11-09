import html
import random
import time
from datetime import datetime

import streamlit as st

from Retrieval import (
    vectorstore,
    llm,
    rag_prompt,
    general_prompt,
)

st.set_page_config(
    page_title="NyayaAI - Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    --gold: #e0b45a;
    --gold-soft: rgba(224, 180, 90, 0.15);
    --navy-900: #070b1a;
    --navy-800: #0d1330;
    --navy-700: #141b40;
    --text: #e8ecf8;
    --muted: #9aa3c4;
    --teal: #3ddbc0;
    --violet: #8b7cff;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(139,124,255,0.20), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(224,180,90,0.14), transparent 55%),
        radial-gradient(700px 600px at 50% 120%, rgba(61,219,192,0.10), transparent 60%),
        linear-gradient(180deg, var(--navy-900), var(--navy-800));
    color: var(--text);
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.block-container { padding-top: 2rem; max-width: 1100px; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.4rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(20,27,64,0.9), rgba(13,19,48,0.7));
    border: 1px solid rgba(224,180,90,0.25);
    box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.6rem;
    animation: fadeUp 0.8s ease both;
}
.hero::before {
    content: "⚖";
    position: absolute;
    right: -10px; top: -50px;
    font-size: 260px;
    color: rgba(224,180,90,0.07);
    transform: rotate(-12deg);
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.05) 50%, transparent 70%);
    background-size: 250% 100%;
    animation: shine 6s linear infinite;
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    background: var(--gold-soft);
    border: 1px solid rgba(224,180,90,0.35);
    margin-bottom: 0.9rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.1;
    background: linear-gradient(90deg, #fff 0%, var(--gold) 55%, #fff4d1 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    margin: 0.7rem 0 0 0;
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 640px;
    font-weight: 300;
}
.pill-row { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.pill {
    padding: 6px 14px;
    font-size: 0.8rem;
    border-radius: 999px;
    color: var(--text);
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
}
.pill b { color: var(--teal); font-weight: 600; }

/* ---------- Suggestion cards / buttons ---------- */
.section-label {
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.4rem 0 0.7rem 0.2rem;
}
div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    width: 100%;
    min-height: 74px;
    text-align: left;
    padding: 0.9rem 1.1rem;
    border-radius: 16px;
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.35;
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    transition: all 0.25s ease;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-4px);
    border-color: var(--gold);
    color: #fff;
    box-shadow: 0 12px 30px rgba(224,180,90,0.18);
    background: linear-gradient(145deg, rgba(224,180,90,0.14), rgba(255,255,255,0.03));
}
div[data-testid="stButton"] > button:active { transform: translateY(-1px); }

/* Compact action buttons (regenerate, etc.) and sidebar buttons */
.st-key-actions button,
section[data-testid="stSidebar"] button {
    min-height: 42px !important;
    padding: 0.4rem 0.9rem !important;
    text-align: center !important;
    font-size: 0.85rem !important;
    border-radius: 12px !important;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--gold); }
div[data-baseweb="tab-highlight"] { background-color: var(--gold) !important; }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    animation: fadeUp 0.5s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(139,124,255,0.18), rgba(139,124,255,0.05));
    border-color: rgba(139,124,255,0.35);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(224,180,90,0.10), rgba(255,255,255,0.03));
    border-color: rgba(224,180,90,0.28);
}

/* ---------- Mode badge ---------- */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 0.6rem;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 500;
}
.mode-badge .dot { width: 8px; height: 8px; border-radius: 50%; }
.mode-rag {
    color: var(--teal);
    background: rgba(61,219,192,0.10);
    border: 1px solid rgba(61,219,192,0.35);
}
.mode-rag .dot { background: var(--teal); box-shadow: 0 0 10px var(--teal); }
.mode-gen {
    color: #ffb86b;
    background: rgba(255,184,107,0.10);
    border: 1px solid rgba(255,184,107,0.35);
}
.mode-gen .dot { background: #ffb86b; box-shadow: 0 0 10px #ffb86b; }
.elapsed { margin-left: 10px; font-size: 0.74rem; color: var(--muted); }

/* ---------- Source cards ---------- */
.source-card {
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-left: 3px solid var(--gold);
}
.source-top { display: flex; justify-content: space-between; gap: 1rem; align-items: center; }
.source-file { font-weight: 600; color: var(--text); font-size: 0.9rem; word-break: break-all; }
.source-page {
    font-size: 0.74rem; color: var(--gold);
    background: var(--gold-soft);
    padding: 2px 10px; border-radius: 999px; white-space: nowrap;
}
.source-meta { font-size: 0.74rem; color: var(--muted); margin-top: 6px; }
.bar {
    height: 6px; border-radius: 999px; margin-top: 8px;
    background: rgba(255,255,255,0.08); overflow: hidden;
}
.bar > span {
    display: block; height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--violet), var(--teal));
}

/* ---------- Expander / sidebar ---------- */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f26, #0d1330);
    border-right: 1px solid rgba(224,180,90,0.18);
}
.side-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #fff, var(--gold));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.side-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 1rem; }
.stat-card {
    display: flex; align-items: center; gap: 12px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.6rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.stat-ico {
    width: 38px; height: 38px; border-radius: 10px;
    display: grid; place-items: center; font-size: 1.1rem;
    background: var(--gold-soft);
}
.stat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.stat-value { font-size: 0.92rem; font-weight: 600; color: var(--text); }
.disclaimer {
    margin-top: 0.8rem; padding: 0.8rem 0.9rem;
    font-size: 0.78rem; line-height: 1.45;
    color: #ffd9a0;
    background: rgba(255,184,107,0.08);
    border: 1px solid rgba(255,184,107,0.28);
    border-radius: 12px;
}
.hint {
    font-size: 0.78rem; color: var(--muted); margin: 0.2rem 0 0.8rem 0.2rem;
}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    border-radius: 18px;
    border: 1px solid rgba(224,180,90,0.35);
    background: rgba(13,19,48,0.85);
    box-shadow: 0 0 0 4px rgba(224,180,90,0.06);
}
[data-testid="stBottom"] > div { background: transparent; }

/* ---------- Contrast fixes: force light text on the dark background ---------- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stRadio"] label, [data-testid="stRadio"] label * {
    color: var(--text);
}
.hero h1 { -webkit-text-fill-color: transparent; color: transparent; }
.side-brand { -webkit-text-fill-color: transparent; color: transparent; }
.hero-badge, .source-page { color: var(--gold) !important; }
.hero p, .side-sub, .stat-label, .section-label, .hint, .source-meta, .elapsed,
[data-testid="stMetricLabel"] * { color: var(--muted) !important; }
.mode-rag, .mode-rag * { color: var(--teal) !important; }
.mode-gen, .mode-gen * { color: #ffb86b !important; }
.disclaimer, .disclaimer * { color: #ffd9a0 !important; }
.pill b { color: var(--teal) !important; }

/* Slider value labels and tick text */
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--gold) !important; }

/* Expander headers (were white) */
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 14px;
}
[data-testid="stExpander"] summary:hover { background: rgba(224,180,90,0.10) !important; }
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] svg { color: var(--text) !important; fill: var(--text); }

/* Status box ("Found 5 relevant passages") */
[data-testid="stStatusWidget"], [data-testid="stStatus"] details {
    background: rgba(61,219,192,0.08) !important;
    border: 1px solid rgba(61,219,192,0.35) !important;
    border-radius: 14px;
}

/* Chat input: dark box with light text */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: rgba(13,19,48,0.95) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; opacity: 1; }
[data-testid="stChatInput"] button { background: var(--gold) !important; }
[data-testid="stChatInput"] button svg { color: #1a1300 !important; fill: #1a1300 !important; }

/* Tooltips / toasts / alerts */
[data-testid="stToast"], [data-testid="stAlert"] {
    background: var(--navy-700) !important;
    border: 1px solid rgba(224,180,90,0.35);
}

/* Header toolbar (Stop / Deploy) */
[data-testid="stToolbar"] * { color: var(--text) !important; }

/* ---------- Animations ---------- */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shine {
    0%   { background-position: 200% 0; }
    100% { background-position: -50% 0; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODE_AUTO = "🤖 Auto (recommended)"
MODE_DOCS = "📚 Documents only"
MODE_GENERAL = "🌐 General knowledge only"

SUGGESTION_TABS = {
    "📜 Constitution": [
        "What are the fundamental rights guaranteed under the Constitution of India?",
        "What are the Directive Principles of State Policy?",
    ],
    "🔍 Criminal law": [
        "Explain the procedure for filing an FIR.",
        "What is the difference between bailable and non-bailable offences?",
    ],
    "🤝 Civil & contracts": [
        "What are the essential elements of a valid contract?",
        "What remedies are available for breach of contract?",
    ],
}


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("k", 5)
    st.session_state.setdefault("threshold", 1.0)
    st.session_state.setdefault("answer_mode", MODE_AUTO)


init_state()


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def clear_conversation():
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.toast("Conversation cleared", icon="🗑️")


def ask(text):
    st.session_state.pending_question = text


def regenerate():
    """Remove the last answer and ask the same question again."""
    msgs = st.session_state.messages
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
        msgs.pop()
        st.session_state.pending_question = msgs.pop()["content"]
        st.toast("Regenerating answer...", icon="🔄")


def reset_settings():
    st.session_state.k = 5
    st.session_state.threshold = 1.0
    st.session_state.answer_mode = MODE_AUTO
    st.toast("Settings reset", icon="↩️")


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def mode_badge(mode, elapsed=None):
    css = "mode-rag" if mode.startswith("RAG") else "mode-gen"
    timing = f'<span class="elapsed">⏱ {elapsed:.1f}s</span>' if elapsed else ""
    return (
        f'<span class="mode-badge {css}">'
        f'<span class="dot"></span>{html.escape(mode)}</span>{timing}'
    )


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 View source documents ({len(sources)})"):
        for source in sources:
            limit = source.get("threshold", 1.0) or 1.0
            relevance = max(0.0, min(1.0, 1 - source["distance"] / limit))
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-top">
                        <span class="source-file">📄 {html.escape(str(source['file']))}</span>
                        <span class="source-page">Page {html.escape(str(source['page']))}</span>
                    </div>
                    <div class="source-meta">
                        Distance {source['distance']:.4f} · Relevance {relevance:.0%}
                    </div>
                    <div class="bar"><span style="width:{relevance * 100:.0f}%"></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def chat_to_text():
    lines = [f"NyayaAI conversation — {datetime.now():%d %b %Y, %H:%M}", "=" * 50, ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "NyayaAI"
        lines.append(f"{who}:\n{m['content']}\n")
        if m.get("sources"):
            lines.append("Sources:")
            for s in m["sources"]:
                lines.append(f"  - {s['file']} (page {s['page']})")
            lines.append("")
    return "\n".join(lines)


def render_message(index, message, is_last):
    avatar = "🧑‍💼" if message["role"] == "user" else "⚖️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] != "assistant":
            return

        if message.get("mode"):
            st.markdown(
                mode_badge(message["mode"], message.get("elapsed")),
                unsafe_allow_html=True,
            )

        render_sources(message.get("sources"))

        # Quick feedback
        if hasattr(st, "feedback"):
            st.feedback("thumbs", key=f"feedback_{index}")

        # Actions on the latest answer
        if is_last:
            with st.container(key="actions"):
                c1, c2, _ = st.columns([1, 1, 3])
                c1.button(
                    "🔄 Regenerate",
                    key="regen",
                    on_click=regenerate,
                    help="Ask the same question again",
                )
                c2.button(
                    "✏️ New topic",
                    key="newtopic",
                    on_click=clear_conversation,
                    help="Start a fresh conversation",
                )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand">⚖️ NyayaAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-sub">AI-powered legal document search and question answering.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Settings", expanded=True):
        st.radio(
            "Answer source",
            [MODE_AUTO, MODE_DOCS, MODE_GENERAL],
            key="answer_mode",
            help=(
                "Auto: use your documents when relevant, otherwise general "
                "knowledge. Documents only: never answer without a matching "
                "document. General: skip document search."
            ),
        )
        st.slider(
            "Passages to retrieve",
            1, 10,
            key="k",
            help="How many document passages are searched for each question.",
        )
        st.slider(
            "Match strictness (max distance)",
            0.3, 2.0,
            step=0.05,
            key="threshold",
            help=(
                "Lower = only very close matches are used. "
                "Higher = more passages are accepted."
            ),
        )
        st.button("↩️ Reset settings", on_click=reset_settings)

    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    doc_answers = sum(
        1 for m in st.session_state.messages
        if m["role"] == "assistant" and m.get("sources")
    )
    c1, c2 = st.columns(2)
    c1.metric("Questions", user_turns)
    c2.metric("From documents", doc_answers)

    st.markdown(
        """
        <div class="stat-card"><div class="stat-ico">📚</div><div>
            <div class="stat-label">Knowledge base</div>
            <div class="stat-value">Your legal PDFs</div></div></div>
        <div class="stat-card"><div class="stat-ico">🧭</div><div>
            <div class="stat-label">Retrieval</div>
            <div class="stat-value">FAISS</div></div></div>
        <div class="stat-card"><div class="stat-ico">🧠</div><div>
            <div class="stat-label">Language model</div>
            <div class="stat-value">Llama 3.2 · Ollama</div></div></div>
        <div class="disclaimer">
            ⚠️ General answers are not verified against the legal database.
            This tool is for research only and is not legal advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.session_state.messages:
        st.download_button(
            "⬇️ Download chat",
            data=chat_to_text(),
            file_name="nyayaai_conversation.txt",
            mime="text/plain",
        )
    st.button("🗑️ Clear conversation", on_click=clear_conversation)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">Indian Legal Research Assistant</span>
        <h1>NyayaAI</h1>
        <p>Ask questions about the legal documents in your database, or any
        general question. Every answer shows whether it came from your
        documents or from the model's own knowledge.</p>
        <div class="pill-row">
            <span class="pill">🔎 <b>Semantic</b> search</span>
            <span class="pill">📄 <b>Cited</b> sources</span>
            <span class="pill">🔒 <b>Local</b> LLM</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Starter questions (only on an empty chat)
# ----------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<div class="section-label">✨ Try asking</div>', unsafe_allow_html=True)

    tabs = st.tabs(list(SUGGESTION_TABS.keys()))
    for t, (tab_name, questions) in zip(tabs, SUGGESTION_TABS.items()):
        with t:
            cols = st.columns(2)
            for i, text in enumerate(questions):
                cols[i % 2].button(
                    text,
                    key=f"suggest_{tab_name}_{i}",
                    on_click=ask,
                    args=(text,),
                )

    all_questions = [q for qs in SUGGESTION_TABS.values() for q in qs]
    st.button(
        "🎲 Surprise me",
        key="surprise",
        on_click=lambda: ask(random.choice(all_questions)),
    )
    st.markdown(
        '<div class="hint">💡 Tip: use the ⚙️ Settings in the sidebar to control '
        "how strictly documents are matched.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
messages = st.session_state.messages
for i, message in enumerate(messages):
    render_message(i, message, is_last=(i == len(messages) - 1))


# ----------------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------------
def retrieve(question, k, threshold):
    results = vectorstore.similarity_search_with_score(question, k=k)
    # Assumes the existing index uses L2 distance.
    return [
        (doc, float(distance))
        for doc, distance in results
        if float(distance) <= threshold
    ]


def stream_chain(chain, inputs):
    for chunk in chain.stream(inputs):
        yield getattr(chunk, "content", str(chunk))


def build_context(relevant_docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc, _ in relevant_docs
    )


def answer_question(question):
    """Runs retrieval + streaming generation. Returns (answer, mode, sources)."""
    choice = st.session_state.answer_mode
    k = st.session_state.k
    threshold = st.session_state.threshold

    relevant_docs = []
    if choice != MODE_GENERAL:
        with st.status("🔎 Searching the legal database...", expanded=False) as status:
            relevant_docs = retrieve(question, k, threshold)
            if relevant_docs:
                status.update(
                    label=f"✅ Found {len(relevant_docs)} relevant passage(s)",
                    state="complete",
                )
            else:
                status.update(label="No matching passages found", state="complete")

    if relevant_docs:
        stream = stream_chain(
            rag_prompt | llm,
            {"question": question, "context": build_context(relevant_docs)},
        )
        answer = st.write_stream(stream)
        mode = "RAG - Retrieved legal documents"
        sources = [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "distance": distance,
                "threshold": threshold,
            }
            for doc, distance in relevant_docs
        ]
        return answer, mode, sources

    if choice == MODE_DOCS:
        answer = (
            "I couldn't find a relevant passage in your legal documents for "
            "this question. Try rephrasing it, raising the **match strictness** "
            "in the sidebar, or switch the answer source to **Auto**."
        )
        st.markdown(answer)
        return answer, "Documents only - No relevant documents found", []

    answer = st.write_stream(stream_chain(general_prompt | llm, {"question": question}))
    mode = (
        "General LLM - Document search switched off"
        if choice == MODE_GENERAL
        else "General LLM - No relevant documents selected"
    )
    return answer, mode, []


# ----------------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------------
typed = st.chat_input("Ask your legal question...")
question = typed or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    answered = False
    with st.chat_message("assistant", avatar="⚖️"):
        try:
            started = time.time()
            answer, mode, sources = answer_question(question)
            elapsed = time.time() - started

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "mode": mode,
                "sources": sources,
                "elapsed": elapsed,
            })
            answered = True

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")
            st.button(
                "🔄 Try again",
                key="retry",
                on_click=ask,
                args=(question,),
            )
            # Drop the unanswered question so history stays consistent.
            st.session_state.messages.pop()

    if answered:
        # Re-render so the badge, sources, feedback and actions appear.
        st.rerun()