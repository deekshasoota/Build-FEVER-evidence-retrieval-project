"""Streamlit workspace for claim checking, evidence search, and retrieval comparison."""
from pathlib import Path
import gzip
import json
import os
import shutil
import tempfile
from urllib.parse import quote
import streamlit as st

from src.data.load_data import read_jsonl
from src.indexing.inverted_index import InvertedIndex
from src.indexing.sqlite_index import SQLiteEvidenceIndex
from src.pipeline.fact_verifier import verify
from src.retrieval import bm25_retriever, boolean_retriever, tfidf_retriever

st.set_page_config(page_title="FEVER Evidence Lab", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  .stApp {background-image:radial-gradient(ellipse at 3% 10%,rgba(14,165,233,.10),transparent 33%),radial-gradient(ellipse at 94% 32%,rgba(168,85,247,.09),transparent 30%);}
  .block-container {padding-top:2rem; padding-bottom:4rem; max-width:1160px;}
  [data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.16); background:linear-gradient(180deg,rgba(14,165,233,.06),rgba(168,85,247,.035) 48%,rgba(251,146,60,.035));}
  [data-testid="stSidebar"] [data-testid="stMetric"] {padding:.75rem; border:1px solid rgba(56,189,248,.26); border-left:4px solid #38bdf8; border-radius:12px; background:linear-gradient(115deg,rgba(14,165,233,.10),rgba(168,85,247,.06));}
  .hero {position:relative; overflow:hidden; padding:2rem 2.1rem; border-radius:22px; background:linear-gradient(112deg,#10233f 0%,#174ea6 48%,#6841c8 78%,#a13b92 100%); color:#fff; margin:.3rem 0 1.25rem; box-shadow:0 14px 35px rgba(23,78,166,.2);}
  .hero:before {content:""; position:absolute; width:190px; height:190px; right:70px; bottom:-140px; border-radius:50%; background:rgba(45,212,191,.36); filter:blur(2px);}
  .hero:after {content:""; position:absolute; width:250px; height:250px; right:-60px; top:-135px; border:1px solid rgba(255,255,255,.25); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.055),0 0 0 58px rgba(255,255,255,.04);}
  .hero .eyebrow {font-size:.76rem; letter-spacing:.14em; text-transform:uppercase; color:#a5f3fc; font-weight:700; margin:0 0 .5rem;}
  .hero h1 {font-size:2.15rem; line-height:1.15; margin:0 0 .55rem; color:#fff; letter-spacing:-.035em;}
  .hero p {max-width:690px; margin:0; color:#dbeafe; font-size:1.02rem; line-height:1.55;}
  .section-kicker {font-size:.76rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:#0e9f9a; margin:.5rem 0 .25rem;}
  .st-key-search_panel {padding:1.2rem 1.3rem .25rem; border:1px solid rgba(45,212,191,.34); border-radius:18px; background:linear-gradient(135deg,rgba(14,165,233,.075),rgba(168,85,247,.055) 55%,rgba(251,146,60,.055)); margin:.5rem 0 1.2rem; box-shadow:0 8px 24px rgba(14,165,233,.06);}
  .st-key-example_0 button {border-color:rgba(56,189,248,.58); background:rgba(14,165,233,.12);}
  .st-key-example_1 button {border-color:rgba(45,212,191,.58); background:rgba(20,184,166,.12);}
  .st-key-example_2 button {border-color:rgba(251,146,60,.62); background:rgba(249,115,22,.12);}
  .st-key-example_0 button:hover,.st-key-example_1 button:hover,.st-key-example_2 button:hover {filter:brightness(1.13); border-color:rgba(255,255,255,.45);}
  div[data-testid="stFormSubmitButton"] button {min-height:2.8rem; font-weight:800; border:0; border-radius:10px; color:white; background:linear-gradient(100deg,#0284c7,#4f46e5 55%,#9333ea); box-shadow:0 6px 18px rgba(79,70,229,.25);}
  div[data-testid="stFormSubmitButton"] button:hover {filter:brightness(1.1); border:0; color:white;}
  div[data-testid="stButton"] button {border-radius:999px; transition:transform .15s ease,filter .15s ease;}
  div[data-testid="stButton"] button:hover {transform:translateY(-1px);}
  [class*="st-key-evidence_"] {border-color:rgba(45,212,191,.35)!important; background:linear-gradient(110deg,rgba(14,165,233,.055),rgba(168,85,247,.04)); box-shadow:0 5px 18px rgba(15,23,42,.06);}
  [data-testid="stAlert"] {border-radius:13px;}
  @media (max-width: 700px) {.block-container {padding-top:1rem;} .hero {padding:1.4rem; border-radius:16px;} .hero h1 {font-size:1.7rem;}}
</style>
""", unsafe_allow_html=True)

SQLITE_INDEX_PATH = Path("artifacts/indexes/fever_index.sqlite3")
DEMO_INDEX_PATH = Path("data/demo/fever_demo.sqlite3")
DEMO_INDEX_ARCHIVE_PATH = Path("data/demo/fever_demo.sqlite3.gz")
JSON_INDEX_PATH = Path("artifacts/indexes/fever_index.json")
configured_index = os.getenv("FEVER_INDEX_PATH")
if configured_index:
    INDEX_PATH = Path(configured_index)
elif SQLITE_INDEX_PATH.exists():
    INDEX_PATH = SQLITE_INDEX_PATH
elif DEMO_INDEX_PATH.exists():
    INDEX_PATH = DEMO_INDEX_PATH
elif DEMO_INDEX_ARCHIVE_PATH.exists():
    # Keep the GitHub deploy artifact small; expand the compressed SQLite file
    # into the runtime's temporary directory before opening it.
    expanded_demo_path = Path(tempfile.gettempdir()) / "fever_demo.sqlite3"
    if not expanded_demo_path.exists() or expanded_demo_path.stat().st_mtime < DEMO_INDEX_ARCHIVE_PATH.stat().st_mtime:
        with gzip.open(DEMO_INDEX_ARCHIVE_PATH, "rb") as compressed, expanded_demo_path.open("wb") as expanded:
            shutil.copyfileobj(compressed, expanded)
    INDEX_PATH = expanded_demo_path
elif JSON_INDEX_PATH.exists():
    INDEX_PATH = JSON_INDEX_PATH
else:
    INDEX_PATH = None
INDEX_META_PATH = (
    Path("data/demo/fever_demo.meta.json")
    if INDEX_PATH == Path(tempfile.gettempdir()) / "fever_demo.sqlite3"
    else INDEX_PATH.with_suffix(".meta.json") if INDEX_PATH else None
)
MODEL_PATH = Path("artifacts/models/verifier.pkl")

@st.cache_resource(show_spinner="Loading the knowledge index…")
def load_index(path: str, modified_ns: int) -> InvertedIndex | SQLiteEvidenceIndex:
    if path.endswith(".sqlite3"):
        return SQLiteEvidenceIndex(path)
    return InvertedIndex.load(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import pickle
    with Path(path).open("rb") as stream:
        return pickle.load(stream)

def show_evidence(rows: list[dict], key_prefix: str = "evidence") -> None:
    if not rows:
        st.info("No evidence sentences were retrieved for this query.")
        return
    for row in rows:
        page = row.get("page", "Unknown page")
        sentence_id = row.get("sentence_id", "?")
        score = row.get("score")
        score_label = f"Score {score:.6f}" if isinstance(score, (int, float)) else "Boolean match"
        # Each Streamlit container key must be unique across the whole script
        # run. Retriever comparisons render several ranked lists side by side.
        with st.container(border=True, key=f"{key_prefix}_{row.get('rank', 'unknown')}"):
            meta, score_col = st.columns([5, 1])
            page_title = page.replace("_", " ")
            article_url = f"https://en.wikipedia.org/wiki/{quote(page.replace(' ', '_'))}"
            meta.markdown(f"[{page_title}](<{article_url}>)  ·  Sentence {sentence_id}")
            score_col.caption(f"Rank **#{row.get('rank', '–')}**")
            st.write(row.get("text", ""))
            st.caption(score_label)

def use_example(query: str) -> None:
    st.session_state["claim_input"] = query

st.markdown("""
<div class='hero'>
  <p class='eyebrow'>FEVER · Evidence retrieval and fact checking</p>
  <h1>Explore the evidence behind a claim</h1>
  <p>Search Wikipedia sentences, compare retrieval methods, and inspect the sources before drawing a conclusion.</p>
</div>
""", unsafe_allow_html=True)
if INDEX_PATH:
    index = load_index(str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns)
    index_metadata = json.loads(INDEX_META_PATH.read_text(encoding="utf-8")) if INDEX_META_PATH and INDEX_META_PATH.exists() else {}
else:
    # Keep a clean-checkout deployment usable while making the tiny fixture
    # unmistakably different from a real FEVER corpus index.
    index = InvertedIndex()
    for record in read_jsonl("data/sample/evidence.jsonl"):
        index.add(record)
    index_metadata = {"source": "data/sample/evidence.jsonl", "limit": index.document_count}
sample_index = index.is_sample
deployment_demo = bool(index_metadata.get("deployment_demo"))
with st.sidebar:
    st.markdown("### 🔎 FEVER Lab")
    st.caption("Information retrieval workspace")
    # Evidence search is the useful default when the optional verifier has not
    # been trained yet. The verifier intentionally abstains without that model.
    workspace = st.radio("Choose a workspace", ["Search evidence", "Verify a claim", "Compare retrievers"], index=0)
    top_k = st.slider("Evidence sentences", 1, 20, 5)
    method = st.selectbox("Ranking method", ["BM25", "TF-IDF rerank", "Boolean"]) if workspace == "Search evidence" else "BM25"
    st.divider()
    st.metric("Evidence sentences", f"{index.document_count:,}")
    if index.term_count is not None:
        st.metric("Indexed terms", f"{index.term_count:,}")
    elif isinstance(index, SQLiteEvidenceIndex):
        st.caption("Search index: SQLite FTS5 / BM25")
    st.caption("Verifier: trained baseline available" if MODEL_PATH.exists() else "Verifier: retrieval-only abstention")
    if sample_index:
        st.warning("Demo corpus loaded. These illustrative sentences are not the official FEVER dataset.")

st.markdown("<p class='section-kicker'>Start with a claim or topic</p>", unsafe_allow_html=True)
st.caption("Results are source sentences ranked by text relevance. A high rank does not by itself prove a claim.")
if sample_index:
    st.info(
        f"The current index contains only {index.document_count:,} illustrative example sentences, "
        "not the full FEVER/Wikipedia corpus. Try **Pacific Ocean** to see a sample search result. "
        "Other topics may have no matching evidence until you build an index from the official Wikipedia sentence data."
    )
elif deployment_demo:
    st.info(
        f"Deployment preview index: {index.document_count:,} sentences sampled from the FEVER Wikipedia corpus. "
        "Searches can miss evidence outside this preview. Your local full index is used automatically when available."
    )
elif index_metadata.get("limit") is not None:
    st.warning(
        f"Partial index: {index.document_count:,} sentences were indexed from the FEVER Wikipedia corpus. "
        "Searches for topics outside this portion may return no evidence. A complete index requires processing the full corpus."
    )
if workspace == "Verify a claim" and not MODEL_PATH.exists():
    st.info(
        "No verifier model has been trained yet, so this workspace will always return "
        "**NOT ENOUGH INFO**. It still shows retrieved sentences when available. "
        "Use **Search evidence** to see search results without a verdict."
    )
with st.container(key="search_panel"):
    st.markdown("**Try an example**")
    example_cols = st.columns(3)
    examples = [
        "The Eiffel Tower is located in Paris.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Barack Obama was born in Hawaii.",
    ]
    for col, example in zip(example_cols, examples):
        with col:
            st.button(example, key=f"example_{examples.index(example)}", on_click=use_example, args=(example,))
    with st.form("claim_form", clear_on_submit=False):
        claim = st.text_area("Claim or search query", placeholder="Type a person, place, event, or a full claim…", height=105, key="claim_input")
        submitted = st.form_submit_button("Search evidence  →", type="primary", use_container_width=True)

if submitted:
    if not claim.strip():
        st.error("Enter a claim or search query first.")
    elif workspace == "Verify a claim":
        with st.spinner("Retrieving and reviewing evidence…"):
            result = verify(claim.strip(), index, top_k)
        verdict = result["verdict"]
        if verdict == "SUPPORTS": st.success(f"Verdict: {verdict}")
        elif verdict == "REFUTES": st.error(f"Verdict: {verdict}")
        else: st.warning(f"Verdict: {verdict}")
        st.write(result["explanation"])
        st.caption(f"Method: {result['method']}")
        st.subheader("Retrieved evidence")
        show_evidence(result["evidence"], key_prefix="verified_evidence")
    elif workspace == "Search evidence":
        with st.spinner("Searching the index…"):
            if method == "BM25":
                rows = bm25_retriever.search(claim.strip(), index, top_k)
            elif method == "TF-IDF rerank":
                candidates = bm25_retriever.search(claim.strip(), index, max(100, top_k * 20))
                rows = tfidf_retriever.search(claim.strip(), candidates, top_k)
            else:
                ids = boolean_retriever.search(claim.strip(), index)
                rows = [{**row, "rank": rank, "score": None}
                        for rank, row in enumerate(index.get_documents(ids[:top_k]), 1)]
        st.markdown("<p class='section-kicker'>Search results</p>", unsafe_allow_html=True)
        st.subheader(f"{len(rows)} ranked evidence sentence(s)")
        st.caption(f"Query: {claim.strip()} · Ranking: {method}")
        show_evidence(rows, key_prefix="search_evidence")
    else:
        with st.spinner("Comparing retrieval methods…"):
            bm25_rows = bm25_retriever.search(claim.strip(), index, top_k)
            candidate_rows = bm25_retriever.search(claim.strip(), index, max(100, top_k * 20))
            tfidf_rows = tfidf_retriever.search(claim.strip(), candidate_rows, top_k)
            boolean_ids = boolean_retriever.search(claim.strip(), index)[:top_k]
        left, right = st.columns(2)
        with left:
            st.subheader("BM25")
            show_evidence(bm25_rows, key_prefix="compare_bm25")
        with right:
            st.subheader("TF-IDF over BM25 candidates")
            show_evidence(tfidf_rows, key_prefix="compare_tfidf")
        with st.expander("Boolean matches"):
            show_evidence([{**row, "rank": rank, "score": None}
                           for rank, row in enumerate(index.get_documents(boolean_ids), 1)],
                          key_prefix="compare_boolean")

with st.expander("How to interpret results", expanded=False):
    st.markdown("""
    - **SUPPORTS / REFUTES** are predictions from the saved trained baseline; inspect the evidence sentences before relying on them.
    - **NOT ENOUGH INFO** means this system did not establish a verdict from its current evidence. It does not mean the claim is false.
    - BM25 and TF-IDF scores rank text matches. They are not probabilities or proof.
    - The comparison view reranks BM25 candidate sentences with TF-IDF so it does not vectorize the entire Wikipedia corpus on every query.
    """)

st.caption("Wikipedia text is from the FEVER June 2017 snapshot. [FEVER dataset](https://fever.ai/dataset/fever.html) · [Attribution and reuse terms](https://fever.ai/download/fever/license.html).")
