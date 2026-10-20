"""Streamlit workspace for claim checking, evidence search, and retrieval comparison."""
from pathlib import Path
import json
import streamlit as st

from src.indexing.inverted_index import InvertedIndex
from src.indexing.sqlite_index import SQLiteEvidenceIndex
from src.pipeline.fact_verifier import verify
from src.retrieval import bm25_retriever, boolean_retriever, tfidf_retriever

st.set_page_config(page_title="FEVER Evidence Lab", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1180px;}
  .hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: linear-gradient(115deg,#172554,#1d4ed8); color: white; margin-bottom: 1rem;}
  .hero h1 {margin: 0 0 .35rem 0; color: white;}
  .hero p {margin: 0; color: #dbeafe;}
  .evidence {padding: 1rem 1.15rem; margin: .65rem 0; border: 1px solid rgba(128,128,128,.28); border-radius: 12px;}
  .evidence small {opacity:.78;}
</style>
""", unsafe_allow_html=True)

SQLITE_INDEX_PATH = Path("artifacts/indexes/fever_index.sqlite3")
JSON_INDEX_PATH = Path("artifacts/indexes/fever_index.json")
INDEX_PATH = SQLITE_INDEX_PATH if SQLITE_INDEX_PATH.exists() else JSON_INDEX_PATH
INDEX_META_PATH = INDEX_PATH.with_suffix(".meta.json")
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

def show_evidence(rows: list[dict]) -> None:
    if not rows:
        st.info("No evidence sentences were retrieved for this query.")
        return
    for row in rows:
        page = row.get("page", "Unknown page")
        sentence_id = row.get("sentence_id", "?")
        score = row.get("score")
        score_label = f"Score {score:.6f}" if isinstance(score, (int, float)) else "Boolean match"
        with st.container(border=True):
            st.caption(f"#{row.get('rank', '–')} · {page} · sentence {sentence_id} · {score_label}")
            st.write(row.get("text", ""))

st.markdown("<div class='hero'><h1>FEVER Evidence Lab</h1><p>Find source sentences first. Review the evidence before interpreting a verdict.</p></div>", unsafe_allow_html=True)
if not INDEX_PATH.exists():
    st.warning("The knowledge index is not built yet. Use the disk-based builder on the extracted FEVER wiki-pages folder; see the README for the command.")
    st.code('python -m scripts.build_sqlite_index "/path/to/wiki-pages"', language="bash")
    st.stop()

index = load_index(str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns)
index_metadata = json.loads(INDEX_META_PATH.read_text(encoding="utf-8")) if INDEX_META_PATH.exists() else {}
sample_index = index.is_sample if isinstance(index, SQLiteEvidenceIndex) else index.is_sample
with st.sidebar:
    st.title("FEVER Lab")
    # Evidence search is the useful default when the optional verifier has not
    # been trained yet. The verifier intentionally abstains without that model.
    workspace = st.radio("Workspace", ["Verify a claim", "Search evidence", "Compare retrievers"], index=1)
    top_k = st.slider("Evidence sentences", 1, 20, 5)
    method = st.selectbox("Ranking method", ["BM25", "TF-IDF rerank", "Boolean"]) if workspace == "Search evidence" else "BM25"
    st.divider()
    st.metric("Indexed sentences", f"{index.document_count:,}")
    if index.term_count is not None:
        st.metric("Indexed terms", f"{index.term_count:,}")
    elif isinstance(index, SQLiteEvidenceIndex):
        st.caption("Search index: SQLite FTS5 / BM25")
    st.caption("Verifier: trained baseline available" if MODEL_PATH.exists() else "Verifier: retrieval-only abstention")
    if sample_index:
        st.warning("Demo corpus loaded. These illustrative sentences are not the official FEVER dataset.")

st.caption("Retrieval similarity measures textual relevance. It is not a truth score or calibrated confidence.")
if sample_index:
    st.info(
        f"The current index contains only {index.document_count:,} illustrative example sentences, "
        "not the full FEVER/Wikipedia corpus. Try **Pacific Ocean** to see a sample search result. "
        "Other topics may have no matching evidence until you build an index from the official Wikipedia sentence data."
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
with st.form("claim_form", clear_on_submit=False):
    claim = st.text_area("Claim or search query", placeholder="Example: The Pacific Ocean is the largest ocean on Earth.", height=105)
    submitted = st.form_submit_button("Run", type="primary", use_container_width=True)

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
        show_evidence(result["evidence"])
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
        st.subheader(f"{len(rows)} ranked evidence sentence(s)")
        show_evidence(rows)
    else:
        with st.spinner("Comparing retrieval methods…"):
            bm25_rows = bm25_retriever.search(claim.strip(), index, top_k)
            candidate_rows = bm25_retriever.search(claim.strip(), index, max(100, top_k * 20))
            tfidf_rows = tfidf_retriever.search(claim.strip(), candidate_rows, top_k)
            boolean_ids = boolean_retriever.search(claim.strip(), index)[:top_k]
        left, right = st.columns(2)
        with left:
            st.subheader("BM25")
            show_evidence(bm25_rows)
        with right:
            st.subheader("TF-IDF over BM25 candidates")
            show_evidence(tfidf_rows)
        with st.expander("Boolean matches"):
            show_evidence([{**row, "rank": rank, "score": None}
                           for rank, row in enumerate(index.get_documents(boolean_ids), 1)])

with st.expander("How to interpret results"):
    st.markdown("""
    - **SUPPORTS / REFUTES** are predictions from the saved trained baseline; inspect the evidence sentences before relying on them.
    - **NOT ENOUGH INFO** means this system did not establish a verdict from its current evidence. It does not mean the claim is false.
    - BM25 and TF-IDF scores rank text matches. They are not probabilities or proof.
    - The comparison view reranks BM25 candidate sentences with TF-IDF so it does not vectorize the entire Wikipedia corpus on every query.
    """)
