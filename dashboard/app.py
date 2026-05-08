from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from local.ollama_client import WaterPracticesClient
from local.pipeline import DocumentAnalyzer, DocumentResult, GOOD_CATEGORIES, BAD_CATEGORIES

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="HydroAnalysis · LLM Edition",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS — estética HydroAnalysis (NASA × brutalist)
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

:root {
  --bg-void:       #040810;
  --bg-deep:       #080f1e;
  --bg-panel:      #0c1628;
  --bg-card:       #0f1e35;
  --bg-input:      #091220;
  --accent-cyan:   #00d4ff;
  --accent-green:  #00ff9d;
  --accent-red:    #ff3b5c;
  --accent-amber:  #ffb800;
  --accent-blue:   #4d9fff;
  --text-primary:  #e8f4ff;
  --text-secondary:#7a9bbf;
  --text-muted:    #3d5a7a;
  --border-dim:    rgba(77,159,255,0.12);
  --border-active: rgba(0,212,255,0.4);
}

/* Background void with grid */
.stApp {
  background: var(--bg-void) !important;
  background-image:
    linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
}

/* Main container */
.main .block-container {
  padding-top: 2rem;
  max-width: 1400px;
}

/* Headers */
h1, h2, h3 {
  font-family: 'Courier Prime', 'Courier New', monospace !important;
  color: var(--text-primary) !important;
  letter-spacing: 0.03em;
}
h1 { color: var(--accent-cyan) !important; font-size: 2rem !important; }
h2 { font-size: 1.3rem !important; border-bottom: 1px solid var(--border-dim); padding-bottom: 8px; margin-top: 2rem !important; }

/* Body text */
.stApp, .stMarkdown, p, span, div, label {
  color: var(--text-primary);
  font-family: Georgia, serif;
}

/* Sidebar / drop zone */
.stFileUploader > div {
  background: rgba(15,30,53,0.7) !important;
  border: 2px dashed var(--border-active) !important;
  border-radius: 16px !important;
  padding: 2rem !important;
}

/* Buttons */
.stButton > button {
  background: var(--accent-cyan) !important;
  color: var(--bg-void) !important;
  font-family: 'Courier Prime', monospace !important;
  font-weight: bold !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 4px !important;
  padding: 10px 24px !important;
  transition: all 0.25s !important;
}
.stButton > button:hover {
  box-shadow: 0 0 20px rgba(0,212,255,0.5) !important;
  transform: translateY(-1px);
}

/* Metrics */
[data-testid="stMetricValue"] {
  font-family: 'Courier Prime', monospace !important;
  font-size: 2rem !important;
  color: var(--accent-cyan) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'Courier Prime', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
}

/* Cards / containers */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  border-radius: 8px;
  padding: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: 4px !important;
  font-family: 'Courier Prime', monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,212,255,0.1) !important;
  border-color: var(--accent-cyan) !important;
  color: var(--accent-cyan) !important;
}

/* Status pills */
.status-pill {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-family: 'Courier Prime', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}
.status-good { background: rgba(0,255,157,0.1); color: var(--accent-green); border: 1px solid var(--accent-green); }
.status-bad  { background: rgba(255,59,92,0.1); color: var(--accent-red); border: 1px solid var(--accent-red); }
.status-neutral { background: rgba(255,184,0,0.1); color: var(--accent-amber); border: 1px solid var(--accent-amber); }

/* Highlight spans */
.highlight-good {
  background: rgba(0,255,157,0.15);
  border-left: 3px solid var(--accent-green);
  padding: 8px 12px;
  margin: 8px 0;
  border-radius: 4px;
  font-family: Georgia, serif;
}
.highlight-bad {
  background: rgba(255,59,92,0.15);
  border-left: 3px solid var(--accent-red);
  padding: 8px 12px;
  margin: 8px 0;
  border-radius: 4px;
  font-family: Georgia, serif;
}

/* Progress bar */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border-dim); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="font-family:'Courier Prime',monospace;font-size:0.7rem;letter-spacing:0.3em;color:#00d4ff;text-transform:uppercase;">
        SYSTEM ANALYSIS · v3.0 · LLM-POWERED
    </div>
    <h1 style="margin:0;">HydroAnalysis</h1>
    <div style="font-family:'Trebuchet MS',sans-serif;font-size:0.85rem;letter-spacing:0.05em;color:#7a9bbf;">
        Rural water monitoring practices · Fine-tuned Qwen 2.5 7B · Local inference
    </div>
    """, unsafe_allow_html=True)

with col2:
    # System status
    @st.cache_resource
    def get_client():
        try:
            return WaterPracticesClient(), None
        except Exception as e:
            return None, str(e)

    client, error = get_client()
    if client:
        st.markdown("""
        <div style="font-family:'Courier Prime',monospace;font-size:0.75rem;color:#00ff9d;text-align:right;margin-top:1rem;">
            ● SYSTEM ONLINE<br>
            <span style="color:#3d5a7a;">water-practices · ollama</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-family:'Courier Prime',monospace;font-size:0.75rem;color:#ff3b5c;text-align:right;margin-top:1rem;">
            ● SYSTEM OFFLINE<br>
            <span style="color:#3d5a7a;font-size:0.7rem;">{error[:80]}</span>
        </div>
        """, unsafe_allow_html=True)
        st.error("No se puede conectar al modelo. Verifica que Ollama esté corriendo y que el modelo 'water-practices' esté creado.")
        st.stop()

st.markdown("---")


# ============================================================================
# UPLOAD ZONE
# ============================================================================

st.markdown("## 01 · Document Upload")

uploaded_files = st.file_uploader(
    "Drag and drop PDF or TXT files here",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="You can upload multiple files at once. Each will be analyzed separately.",
)

if uploaded_files:
    st.markdown(f"""
    <div style="font-family:'Courier Prime',monospace;font-size:0.75rem;color:#7a9bbf;letter-spacing:0.15em;">
        {len(uploaded_files)} FILE(S) READY
    </div>
    """, unsafe_allow_html=True)

    # File list
    for f in uploaded_files:
        size_kb = len(f.getvalue()) / 1024
        st.markdown(f"""
        <div style="background:#0f1e35;border:1px solid rgba(77,159,255,0.12);border-radius:8px;padding:8px 16px;margin:4px 0;display:flex;justify-content:space-between;">
            <span style="font-family:'Trebuchet MS',sans-serif;color:#e8f4ff;">{f.name}</span>
            <span style="font-family:'Courier Prime',monospace;font-size:0.7rem;color:#3d5a7a;">{size_kb:.1f} KB</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# ANALYZE
# ============================================================================

# Initialize state
if "results" not in st.session_state:
    st.session_state.results = []

col_a, col_b = st.columns([3, 1])
with col_a:
    analyze_clicked = st.button("⚡ Analyze documents", disabled=not uploaded_files, use_container_width=True)
with col_b:
    if st.button("🗑 Clear results", use_container_width=True):
        st.session_state.results = []
        st.rerun()

if analyze_clicked and uploaded_files:
    analyzer = DocumentAnalyzer(client)
    st.session_state.results = []

    overall_progress = st.progress(0.0, text="Initializing...")
    status_text = st.empty()

    for file_idx, uploaded in enumerate(uploaded_files):
        file_bytes = uploaded.getvalue()
        chunk_progress = st.progress(0.0, text=f"{uploaded.name}")

        def cb(done, total, msg):
            chunk_progress.progress(
                done / max(total, 1),
                text=f"{uploaded.name} · {msg}"
            )

        status_text.markdown(f"""
        <div style="font-family:'Courier Prime',monospace;font-size:0.8rem;color:#00d4ff;">
            ▸ Processing {file_idx + 1}/{len(uploaded_files)}: <strong>{uploaded.name}</strong>
        </div>
        """, unsafe_allow_html=True)

        t0 = time.time()
        try:
            result = analyzer.analyze_file(uploaded.name, file_bytes=file_bytes, progress_callback=cb)
            elapsed = time.time() - t0
            st.session_state.results.append({"result": result, "elapsed": elapsed})
        except Exception as e:
            st.error(f"Error procesando {uploaded.name}: {e}")
            continue

        chunk_progress.empty()
        overall_progress.progress(
            (file_idx + 1) / len(uploaded_files),
            text=f"Completed {file_idx + 1}/{len(uploaded_files)}"
        )

    overall_progress.empty()
    status_text.empty()
    st.success(f"✓ Analysis complete · {len(st.session_state.results)} document(s) processed")


# ============================================================================
# RESULTS
# ============================================================================

results_data = st.session_state.results

if not results_data:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#3d5a7a;font-family:'Courier Prime',monospace;letter-spacing:0.1em;">
        Upload documents and click ANALYZE to see results
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Aggregate stats across all documents
all_results: list[DocumentResult] = [r["result"] for r in results_data]

n_docs = len(all_results)
n_good_docs = sum(1 for r in all_results if r.overall_classification == "good")
n_bad_docs = sum(1 for r in all_results if r.overall_classification == "bad")
n_neutral_docs = sum(1 for r in all_results if r.overall_classification == "neutral")
total_practices_good = sum(r.n_good_practices for r in all_results)
total_practices_bad = sum(r.n_bad_practices for r in all_results)


# ----- Overview metrics -----

st.markdown("## 02 · Global Analysis")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Documents", n_docs)
m2.metric("Good docs", n_good_docs)
m3.metric("Bad docs", n_bad_docs)
m4.metric("Good practices", total_practices_good)
m5.metric("Bad practices", total_practices_bad)


# ----- Charts -----

st.markdown("### Distribution & scores")

c1, c2 = st.columns(2)

with c1:
    # Donut: good/bad/neutral docs
    fig = go.Figure(go.Pie(
        labels=["Good", "Bad", "Neutral"],
        values=[n_good_docs, n_bad_docs, n_neutral_docs],
        hole=0.6,
        marker=dict(colors=["rgba(0,255,157,0.4)", "rgba(255,59,92,0.4)", "rgba(255,184,0,0.4)"],
                    line=dict(color=["#00ff9d", "#ff3b5c", "#ffb800"], width=2)),
    ))
    fig.update_layout(
        title="Document classification",
        title_font=dict(family="Courier Prime, monospace", size=14, color="#00d4ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a9bbf", family="Trebuchet MS"),
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Bar: good_score vs bad_score per document
    df_scores = pd.DataFrame([
        {"doc": r.filename[:25] + ("..." if len(r.filename) > 25 else ""),
         "Good score": r.good_score,
         "Bad score": r.bad_score}
        for r in all_results
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Good", x=df_scores["doc"], y=df_scores["Good score"],
                         marker=dict(color="rgba(0,255,157,0.4)", line=dict(color="#00ff9d", width=1.5))))
    fig.add_trace(go.Bar(name="Bad", x=df_scores["doc"], y=df_scores["Bad score"],
                         marker=dict(color="rgba(255,59,92,0.4)", line=dict(color="#ff3b5c", width=1.5))))
    fig.update_layout(
        title="Practice density (% of chunks)",
        title_font=dict(family="Courier Prime, monospace", size=14, color="#00d4ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a9bbf", family="Trebuchet MS"),
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        barmode="group",
        xaxis=dict(showgrid=False, gridcolor="rgba(77,159,255,0.08)"),
        yaxis=dict(gridcolor="rgba(77,159,255,0.08)", title="%"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ----- Aggregate categories -----

cat_counter: dict[str, int] = {}
for r in all_results:
    for cat, count in r.category_counts.items():
        cat_counter[cat] = cat_counter.get(cat, 0) + count

if cat_counter:
    df_cats = pd.DataFrame([
        {"category": cat, "count": count, "type": "good" if cat in GOOD_CATEGORIES else ("bad" if cat in BAD_CATEGORIES else "other")}
        for cat, count in sorted(cat_counter.items(), key=lambda x: -x[1])
    ])
    fig = px.bar(
        df_cats, x="count", y="category", orientation="h", color="type",
        color_discrete_map={"good": "#00ff9d", "bad": "#ff3b5c", "other": "#ffb800"},
    )
    fig.update_layout(
        title="Categories detected (across all documents)",
        title_font=dict(family="Courier Prime, monospace", size=14, color="#00d4ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a9bbf", family="Trebuchet MS"),
        height=400,
        yaxis=dict(autorange="reversed", gridcolor="rgba(77,159,255,0.08)"),
        xaxis=dict(gridcolor="rgba(77,159,255,0.08)"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ----- Per-document drill-down -----

st.markdown("## 03 · Per-document analysis")

for idx, item in enumerate(results_data):
    r: DocumentResult = item["result"]
    elapsed = item["elapsed"]
    cls = r.overall_classification

    pill_class = f"status-{cls}"
    pill_label = {"good": "✓ GOOD PRACTICES", "bad": "✗ BAD PRACTICES", "neutral": "◌ NEUTRAL"}[cls]

    with st.expander(f"{r.filename}  ·  {r.n_chunks} chunks  ·  {elapsed:.1f}s", expanded=(idx == 0)):
        st.markdown(f'<span class="status-pill {pill_class}">{pill_label}</span>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks with practice", f"{r.n_chunks_with_practice}/{r.n_chunks}")
        c2.metric("Good score", f"{r.good_score}%")
        c3.metric("Bad score", f"{r.bad_score}%")
        c4.metric("Top categories", len(r.category_counts))

        tabs = st.tabs(["Top findings", "Categories", "Full chunks", "Summary"])

        with tabs[0]:
            if not r.top_findings:
                st.info("No specific practices were detected with high confidence in this document.")
            else:
                for f in r.top_findings:
                    cls_html = "highlight-good" if f.type == "good" else "highlight-bad"
                    cats_str = ", ".join(f.categories)
                    st.markdown(f"""
                    <div class="{cls_html}">
                      <div style="font-family:'Courier Prime',monospace;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:{'#00ff9d' if f.type == 'good' else '#ff3b5c'};margin-bottom:6px;">
                        {f.type.upper()} · {cats_str} · confidence {f.confidence:.2f}
                      </div>
                      <div style="font-style:italic;color:#e8f4ff;margin-bottom:6px;">"{f.span[:300]}{'...' if len(f.span) > 300 else ''}"</div>
                      <div style="font-size:0.85rem;color:#7a9bbf;">{f.explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[1]:
            if r.category_counts:
                df = pd.DataFrame([
                    {"category": k, "count": v, "type": "good" if k in GOOD_CATEGORIES else "bad"}
                    for k, v in sorted(r.category_counts.items(), key=lambda x: -x[1])
                ])
                fig = px.bar(df, x="count", y="category", orientation="h", color="type",
                             color_discrete_map={"good": "#00ff9d", "bad": "#ff3b5c"})
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#7a9bbf"),
                    height=300,
                    yaxis=dict(autorange="reversed", gridcolor="rgba(77,159,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(77,159,255,0.08)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categories detected.")

        with tabs[2]:
            st.markdown(f"**Showing chunks with detected practices ({r.n_chunks_with_practice} of {r.n_chunks})**")
            for c in r.chunk_results:
                if not c.contains_practice:
                    continue
                with st.container(border=True):
                    st.markdown(f"<div style='font-family:Courier Prime,monospace;font-size:0.7rem;color:#3d5a7a;letter-spacing:0.1em;'>CHUNK #{c.chunk_index}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#7a9bbf;font-size:0.85rem;margin:6px 0;'>{c.text[:400]}{'...' if len(c.text) > 400 else ''}</div>", unsafe_allow_html=True)
                    for p in c.practices:
                        cls_html = "highlight-good" if p.type == "good" else "highlight-bad"
                        st.markdown(f"""
                        <div class="{cls_html}" style="margin-top:8px;">
                          <strong>{p.type.upper()}</strong> · {", ".join(p.categories)} · {p.confidence:.2f}<br>
                          <em style="font-size:0.85rem;">{p.explanation}</em>
                        </div>
                        """, unsafe_allow_html=True)

        with tabs[3]:
            st.markdown(f"**Document summary** _(generated by aggregating chunk-level summaries)_")
            st.markdown(f"<div style='background:#091220;border-left:3px solid #00d4ff;padding:12px;border-radius:4px;'>{r.document_summary}</div>", unsafe_allow_html=True)


# ============================================================================
# EXPORT
# ============================================================================

if results_data:
    st.markdown("## 04 · Export")

    e1, e2 = st.columns(2)

    with e1:
        # JSON export
        export_data = [r["result"].to_dict() for r in results_data]
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
            file_name="hydroanalysis_results.json",
            mime="application/json",
            use_container_width=True,
        )

    with e2:
        # CSV export — flatten para análisis en Excel
        csv_rows = []
        for r in all_results:
            for c in r.chunk_results:
                for p in c.practices:
                    csv_rows.append({
                        "document": r.filename,
                        "doc_classification": r.overall_classification,
                        "chunk_index": c.chunk_index,
                        "type": p.type,
                        "categories": ";".join(p.categories),
                        "confidence": p.confidence,
                        "span": p.span[:200],
                        "explanation": p.explanation[:200],
                    })
        if csv_rows:
            csv_df = pd.DataFrame(csv_rows)
            st.download_button(
                "⬇ Download CSV (flat)",
                data=csv_df.to_csv(index=False),
                file_name="hydroanalysis_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )
