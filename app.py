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
from src.retrieval import (bm25_retriever, boolean_retriever,
                           embedding_retriever, rrf_retriever, tfidf_retriever)

st.set_page_config(page_title="FEVER Retrieval Study", page_icon="📄", layout="wide")
st.markdown("""
<style>
  :root {--ir-plum:#5b4b8a; --ir-coral:#d7654b; --ir-teal:#258b82; --ir-gold:#d9a62e; --ir-blue:#3976a8;}
  .block-container {padding-top:1.6rem; padding-bottom:3.5rem; max-width:1080px;}
  [data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.18);}
  [data-testid="stSidebar"] [data-testid="stMetric"] {padding:.7rem; border:1px solid rgba(128,128,128,.2); border-left:3px solid var(--ir-teal); border-radius:7px; background:rgba(37,139,130,.07);}
  .hero {position:relative; padding:1.35rem 1.5rem 1.4rem; border:1px solid rgba(128,128,128,.24); border-radius:9px; margin:.25rem 0 1.1rem; background:rgba(217,166,46,.07);}
  .hero:before {content:""; position:absolute; top:0; left:0; right:0; height:4px; border-radius:9px 9px 0 0; background:linear-gradient(90deg,var(--ir-plum) 0 48%,var(--ir-coral) 48% 74%,var(--ir-teal) 74%);}
  .hero .eyebrow {font-size:.78rem; letter-spacing:.07em; text-transform:uppercase; color:var(--ir-plum); font-weight:750; margin:.1rem 0 .55rem;}
  .hero h1 {font-size:1.9rem; line-height:1.2; margin:0 0 .45rem; letter-spacing:-.02em;}
  .hero p {max-width:760px; margin:0; font-size:1rem; line-height:1.55; opacity:.88;}
  .section-kicker {font-size:.8rem; font-weight:750; letter-spacing:.05em; text-transform:uppercase; color:var(--ir-teal); margin:.5rem 0 .25rem;}
  .st-key-search_panel {padding:1rem 1.1rem .25rem; border:1px solid rgba(128,128,128,.22); border-left:4px solid var(--ir-gold); border-radius:8px; margin:.5rem 0 1.1rem; background:rgba(217,166,46,.045);}
  .st-key-example_0 button {border-color:rgba(91,75,138,.52); background:rgba(91,75,138,.09);}
  .st-key-example_1 button {border-color:rgba(37,139,130,.52); background:rgba(37,139,130,.09);}
  .st-key-example_2 button {border-color:rgba(215,101,75,.52); background:rgba(215,101,75,.09);}
  div[data-testid="stFormSubmitButton"] button {min-height:2.65rem; font-weight:700; border-radius:7px; background:var(--ir-plum); border-color:var(--ir-plum);}
  div[data-testid="stFormSubmitButton"] button:hover {background:#493b72; border-color:#493b72;}
  div[data-testid="stButton"] button {border-radius:7px; transition:filter .15s ease;}
  div[data-testid="stButton"] button:hover {filter:brightness(1.08);}
  [class*="st-key-evidence_"] {border-color:rgba(128,128,128,.24)!important; border-radius:7px;}
  [class*="st-key-compare_bm25_"] {border-left:4px solid var(--ir-coral)!important;}
  [class*="st-key-compare_tfidf_"] {border-left:4px solid var(--ir-teal)!important;}
  [class*="st-key-compare_boolean_"] {border-left:4px solid var(--ir-plum)!important;}
  [class*="st-key-compare_embedding_"] {border-left:4px solid var(--ir-blue)!important;}
  [class*="st-key-compare_hybrid_"] {border-left:4px solid var(--ir-gold)!important;}
  .stTabs [data-baseweb="tab-list"] {gap:.35rem;}
  .stTabs [data-baseweb="tab"] {border-radius:7px 7px 0 0; padding:.55rem .8rem;}
  .stTabs [data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"] {border-bottom:3px solid var(--ir-coral);}
  .stTabs [data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"] {border-bottom:3px solid var(--ir-teal);}
  .stTabs [data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"] {border-bottom:3px solid var(--ir-plum);}
  .stTabs [data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"] {border-bottom:3px solid var(--ir-blue);}
  .stTabs [data-baseweb="tab-list"] button:nth-child(5)[aria-selected="true"] {border-bottom:3px solid var(--ir-gold);}
  [data-testid="stAlert"] {border-radius:7px;}
  @media (max-width: 700px) {.block-container {padding-top:1rem;} .hero {padding:1.1rem 1rem;} .hero h1 {font-size:1.55rem;}}
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

def preprocessing_config(index) -> dict:
    return getattr(index, "preprocessing", {"remove_stopwords": False, "stemming": False})

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import pickle
    with Path(path).open("rb") as stream:
        return pickle.load(stream)

@st.cache_resource(show_spinner="Preparing corpus-wide retrieval indexes…")
def load_retrieval_corpus(path: str, modified_ns: int):
    retrieval_index = load_index(path, modified_ns)
    if retrieval_index.document_count > 100_000:
        raise RuntimeError(
            "TF-IDF and dense ranking are limited to 100,000 sentences in the interactive app to avoid excessive memory use. "
            "Use a smaller comparison index."
        )
    return (retrieval_index.all_documents() if isinstance(retrieval_index, SQLiteEvidenceIndex)
            else list(retrieval_index.documents.values()))

@st.cache_resource(show_spinner="Building the corpus-wide TF-IDF matrix…")
def load_tfidf_resources(path: str, modified_ns: int):
    retrieval_index = load_index(path, modified_ns)
    records = load_retrieval_corpus(path, modified_ns)
    vectorizer, matrix = tfidf_retriever.build_index(records, preprocessing_config(retrieval_index))
    return records, vectorizer, matrix

@st.cache_resource(show_spinner="Loading sentence encoder and evidence vectors…")
def load_embedding_resources(path: str, modified_ns: int, model_name: str):
    records = load_retrieval_corpus(path, modified_ns)
    encoder, vectors = embedding_retriever.encode_records(records, model_name)
    return records, encoder, vectors

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
  <p class='eyebrow'>Information retrieval semester project · FEVER</p>
  <h1>Comparing methods for FEVER evidence retrieval</h1>
  <p>This project compares TF-IDF, BM25, sentence embeddings, and reciprocal-rank fusion over Wikipedia sentences associated with FEVER claims. Results are ranked text matches; they are not proof of a claim.</p>
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
    st.markdown("### FEVER Retrieval Study")
    st.caption("TF-IDF · BM25 · Embeddings · RRF")
    # Evidence search is the useful default when the optional verifier has not
    # been trained yet. The verifier intentionally abstains without that model.
    workspace = st.radio("Task", ["Search evidence", "Verify a claim", "Compare methods"], index=0)
    top_k = st.slider("Results per method", 1, 20, 10)
    method = st.selectbox("Ranking method", ["BM25", "TF-IDF", "Sentence Embedding", "Hybrid RRF", "Boolean"]) if workspace == "Search evidence" else "BM25"
    st.divider()
    st.metric("Evidence sentences", f"{index.document_count:,}")
    if index.term_count is not None:
        st.metric("Indexed terms", f"{index.term_count:,}")
    elif isinstance(index, SQLiteEvidenceIndex):
        st.caption("Search index: SQLite FTS5 / BM25")
    st.caption("Verifier: trained baseline available" if MODEL_PATH.exists() else "Verifier: retrieval-only abstention")
    if sample_index:
        st.warning("Demo corpus loaded. These illustrative sentences are not the official FEVER dataset.")

st.markdown("<p class='section-kicker'>Search a claim or topic</p>", unsafe_allow_html=True)
st.caption("Search returns source sentences ranked by textual relevance. Inspect the linked Wikipedia article before drawing a conclusion.")
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
    st.markdown("**Example queries**")
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
            elif method == "TF-IDF":
                try:
                    if INDEX_PATH:
                        records, vectorizer, tfidf_matrix = load_tfidf_resources(
                            str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns,
                        )
                    else:
                        records = list(index.documents.values())
                        vectorizer, tfidf_matrix = tfidf_retriever.build_index(records, preprocessing_config(index))
                    rows = tfidf_retriever.search_index(
                        claim.strip(), records, vectorizer, tfidf_matrix, top_k, preprocessing_config(index),
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    st.error(f"TF-IDF search is unavailable: {exc}")
                    rows = []
            elif method in {"Sentence Embedding", "Hybrid RRF"}:
                try:
                    if INDEX_PATH:
                        records, encoder, vectors = load_embedding_resources(
                            str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns,
                            embedding_retriever.DEFAULT_MODEL,
                        )
                    else:
                        records = list(index.documents.values())
                        encoder, vectors = embedding_retriever.encode_records(records)
                    embedding_rows = embedding_retriever.search(
                        claim.strip(), records, top_k, model=encoder, vectors=vectors,
                    )
                    if method == "Sentence Embedding":
                        rows = embedding_rows
                    else:
                        if INDEX_PATH:
                            records, vectorizer, tfidf_matrix = load_tfidf_resources(
                                str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns,
                            )
                        else:
                            records = list(index.documents.values())
                            vectorizer, tfidf_matrix = tfidf_retriever.build_index(records, preprocessing_config(index))
                        bm25_rows = bm25_retriever.search(claim.strip(), index, top_k)
                        tfidf_rows = tfidf_retriever.search_index(
                            claim.strip(), records, vectorizer, tfidf_matrix, top_k, preprocessing_config(index),
                        )
                        rows = rrf_retriever.fuse([bm25_rows, tfidf_rows, embedding_rows], top_k)
                except Exception as exc:
                    st.error(f"{method} is unavailable: {exc}")
                    rows = []
            else:
                ids = boolean_retriever.search(claim.strip(), index)
                rows = [{**row, "rank": rank, "score": None}
                        for rank, row in enumerate(index.get_documents(ids[:top_k]), 1)]
        st.markdown("<p class='section-kicker'>Search results</p>", unsafe_allow_html=True)
        st.subheader(f"{len(rows)} ranked evidence sentence(s)")
        st.caption(f"Query: {claim.strip()} · Ranking: {method}")
        if method == "Sentence Embedding":
            st.caption(f"Encoder: `{embedding_retriever.DEFAULT_MODEL}` · cosine similarity")
        elif method == "Hybrid RRF":
            st.caption("Fused from BM25, corpus-wide TF-IDF, and sentence embeddings using reciprocal-rank fusion.")
        show_evidence(rows, key_prefix="search_evidence")
    else:
        with st.spinner("Comparing retrieval methods…"):
            bm25_rows = bm25_retriever.search(claim.strip(), index, top_k)
            tfidf_error = None
            try:
                if INDEX_PATH:
                    records, vectorizer, tfidf_matrix = load_tfidf_resources(
                        str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns,
                    )
                else:
                    records = list(index.documents.values())
                    vectorizer, tfidf_matrix = tfidf_retriever.build_index(records, preprocessing_config(index))
                tfidf_rows = tfidf_retriever.search_index(
                    claim.strip(), records, vectorizer, tfidf_matrix, top_k, preprocessing_config(index),
                )
            except (RuntimeError, OSError, ValueError) as exc:
                tfidf_error = str(exc)
                tfidf_rows = []
            boolean_ids = boolean_retriever.search(claim.strip(), index)[:top_k]
            embedding_error = None
            embedding_rows = []
            try:
                if INDEX_PATH:
                    records, encoder, vectors = load_embedding_resources(
                        str(INDEX_PATH), INDEX_PATH.stat().st_mtime_ns,
                        embedding_retriever.DEFAULT_MODEL,
                    )
                else:
                    records = list(index.documents.values())
                    encoder, vectors = embedding_retriever.encode_records(records)
                embedding_rows = embedding_retriever.search(
                    claim.strip(), records, top_k, model=encoder, vectors=vectors,
                )
            except Exception as exc:
                embedding_error = str(exc)
            hybrid_error = tfidf_error or embedding_error
            hybrid_rows = rrf_retriever.fuse([bm25_rows, tfidf_rows, embedding_rows], top_k)
        bm25_tab, tfidf_tab, embedding_tab, hybrid_tab, boolean_tab = st.tabs([
            "BM25 · lexical", "TF-IDF · cosine", "Sentence Embedding", "Hybrid · RRF", "Boolean · term match",
        ])
        with bm25_tab:
            st.caption("Probabilistic term ranking with document-length normalization.")
            show_evidence(bm25_rows, key_prefix="compare_bm25")
        with tfidf_tab:
            st.caption("TF-IDF weights are fitted across the full searchable corpus; cosine similarity ranks the sentences.")
            if tfidf_error:
                st.warning(f"TF-IDF retrieval is unavailable: {tfidf_error}")
            else:
                show_evidence(tfidf_rows, key_prefix="compare_tfidf")
        with embedding_tab:
            st.caption(f"Dense semantic retrieval with `{embedding_retriever.DEFAULT_MODEL}`.")
            if embedding_error:
                st.warning(f"Sentence embedding retrieval is unavailable: {embedding_error}")
            else:
                show_evidence(embedding_rows, key_prefix="compare_embedding")
        with hybrid_tab:
            st.caption("Reciprocal Rank Fusion combines BM25, corpus-wide TF-IDF, and sentence-embedding ranks.")
            if hybrid_error:
                st.warning(f"Hybrid RRF is unavailable: {hybrid_error}")
            else:
                show_evidence(hybrid_rows, key_prefix="compare_hybrid")
        with boolean_tab:
            st.caption("Exact token matches; results are not relevance-ranked.")
            show_evidence([{**row, "rank": rank, "score": None}
                           for rank, row in enumerate(index.get_documents(boolean_ids), 1)],
                          key_prefix="compare_boolean")

with st.expander("How to interpret results", expanded=False):
    st.markdown("""
    - **SUPPORTS / REFUTES** are predictions from the saved trained baseline; inspect the evidence sentences before relying on them.
    - **NOT ENOUGH INFO** means this system did not establish a verdict from its current evidence. It does not mean the claim is false.
    - BM25 and TF-IDF scores rank text matches. They are not probabilities or proof.
    - The comparison view fits one TF-IDF model across the loaded corpus and caches it for reuse.
    """)

st.caption("Wikipedia text is from the FEVER June 2017 snapshot. [FEVER dataset](https://fever.ai/dataset/fever.html) · [Attribution and reuse terms](https://fever.ai/download/fever/license.html).")
