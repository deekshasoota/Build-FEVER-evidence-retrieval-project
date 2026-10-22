import json
import gzip
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.load_data import load_claims
from src.data.validate_data import summarize_claims


@st.cache_data(show_spinner="Loading evidence corpus overview…")
def load_evidence_overview(index_path: str, compressed: bool):
    source = Path(index_path)
    if compressed:
        database = Path(tempfile.gettempdir()) / "fever_dataset_explorer.sqlite3"
        if not database.exists() or database.stat().st_mtime_ns < source.stat().st_mtime_ns:
            with gzip.open(source, "rb") as zipped, database.open("wb") as target:
                shutil.copyfileobj(zipped, target)
    else:
        database = source
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30) as connection:
        sentence_count = connection.execute("SELECT count(*) FROM evidence_fts").fetchone()[0]
        page_rows = connection.execute(
            "SELECT page, count(*) AS sentences FROM evidence_fts GROUP BY page ORDER BY sentences DESC LIMIT 15"
        ).fetchall()
        page_count = connection.execute("SELECT count(DISTINCT page) FROM evidence_fts").fetchone()[0]
    return sentence_count, page_count, page_rows


st.title("Dataset Explorer")
st.caption("Explore claim labels and evidence annotations. Gold annotations are supplied by the dataset; they are not model predictions.")

# Show the indexed evidence collection even when the user has not supplied a claims file.
evidence_meta_path = Path("data/demo/fever_demo.meta.json")
evidence_db_path = Path("data/demo/fever_demo.sqlite3")
evidence_archive_path = Path("data/demo/fever_demo.sqlite3.gz")
if evidence_meta_path.exists() and (evidence_db_path.exists() or evidence_archive_path.exists()):
    try:
        corpus_meta = json.loads(evidence_meta_path.read_text(encoding="utf-8"))
        corpus_path = evidence_db_path if evidence_db_path.exists() else evidence_archive_path
        actual_sentences, page_count, page_rows = load_evidence_overview(
            str(corpus_path), corpus_path.suffix == ".gz"
        )
        st.subheader("Evidence corpus")
        st.caption(f"Source: {corpus_meta.get('source', 'FEVER evidence index')} · Deployment preview")
        corpus_cols = st.columns(4)
        corpus_cols[0].metric("Indexed sentences", f"{actual_sentences:,}")
        corpus_cols[1].metric("Wikipedia pages", f"{page_count:,}")
        corpus_cols[2].metric("Full source size", f"{corpus_meta.get('source_sentence_count', actual_sentences):,} sentences")
        corpus_cols[3].metric("Preview coverage", f"{actual_sentences / max(corpus_meta.get('source_sentence_count', actual_sentences), 1):.2%}")
        st.caption("The hosted app uses a small reproducible preview sample. The full FEVER collection is available for local indexing; preview charts describe only the indexed sample.")
        page_frame = pd.DataFrame(page_rows, columns=["Wikipedia page", "Sentences"])
        if not page_frame.empty:
            page_frame["Wikipedia page"] = page_frame["Wikipedia page"].astype(str).str.replace("_", " ", regex=False)
            page_frame = page_frame.sort_values("Sentences", ascending=True)
            page_figure = px.bar(
                page_frame, x="Sentences", y="Wikipedia page", orientation="h",
                color="Sentences", color_continuous_scale="Tealgrn",
                title="Pages with the most indexed evidence sentences",
            )
            page_figure.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10), height=520)
            st.plotly_chart(page_figure, use_container_width=True, key="dataset_evidence_pages")
    except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        st.warning(f"Could not read the evidence preview index: {exc}")

default_path = Path("data/raw/dev.jsonl")
sample_path = Path("data/sample/claims.jsonl")
initial_path = str(default_path if default_path.exists() else sample_path)
path_text = st.text_input("Claims JSONL path", initial_path, help="Use a local FEVER-format claims file, or upload one below.")
uploaded = st.file_uploader("Or upload a claims JSONL file", type=["jsonl", "json", "txt"], key="dataset_claims_upload")

claims = []
invalid_lines = 0
source_label = ""
if uploaded is not None:
    source_label = uploaded.name
    seen_ids = set()
    try:
        for line in uploaded.getvalue().decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("claim"), str):
                invalid_lines += 1
                continue
            row_id = row.get("id", len(claims) + 1)
            if row_id in seen_ids:
                invalid_lines += 1
                continue
            seen_ids.add(row_id)
            row.setdefault("id", row_id)
            claims.append(row)
    except UnicodeDecodeError:
        st.error("This file is not valid UTF-8 JSONL. Export the claims file as UTF-8 and upload it again.")
elif Path(path_text).is_file():
    claims = load_claims(path_text)
    source_label = Path(path_text).name
else:
    st.info("No claims file was found. The bundled example is selected by default; upload your labeled claims file to explore the real dataset.")

if claims:
    is_example = Path(path_text) == sample_path and uploaded is None
    if is_example:
        st.info("Showing the bundled example claims. This tiny fixture demonstrates the page only; it has no gold evidence annotations and must not be used as benchmark results.")
    st.caption(f"Source: {source_label} · {len(claims):,} claim records")
    if invalid_lines:
        st.warning(f"Skipped {invalid_lines:,} malformed or duplicate records while reading the uploaded file.")

    summary = summarize_claims(claims)
    labels = summary["labels"]
    evidence_claims = summary["with_evidence"]
    evidence_sentence_counts = []
    evidence_group_counts = []
    claim_table = []
    for row in claims:
        evidence = row.get("evidence") or []
        if isinstance(evidence, list) and evidence and isinstance(evidence[0], list):
            groups = [group for group in evidence if isinstance(group, list)]
            sentence_count = sum(len(group) for group in groups)
            group_count = len(groups)
        elif isinstance(evidence, list):
            sentence_count = len(evidence)
            group_count = int(bool(evidence))
        else:
            sentence_count = 0
            group_count = 0
        evidence_sentence_counts.append(sentence_count)
        evidence_group_counts.append(group_count)
        claim_table.append({
            "ID": row.get("id"),
            "Claim": row.get("claim", ""),
            "Label": row.get("label") or "UNLABELED",
            "Evidence groups": group_count,
            "Annotated evidence sentences": sentence_count,
        })

    metric_cols = st.columns(4)
    metric_cols[0].metric("Claims", f"{summary['records']:,}")
    metric_cols[1].metric("With evidence", f"{evidence_claims:,}")
    metric_cols[2].metric("Without evidence", f"{summary['records'] - evidence_claims:,}")
    metric_cols[3].metric("Labels", f"{len(labels):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Gold label distribution")
        label_frame = pd.DataFrame([{"Label": str(label), "Claims": count} for label, count in sorted(labels.items())])
        if not label_frame.empty:
            figure = px.bar(label_frame, x="Label", y="Claims", color="Label", text="Claims", title="Claims by gold label")
            figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(figure, use_container_width=True, key="dataset_label_distribution")
    with right:
        st.subheader("Evidence annotation coverage")
        evidence_frame = pd.DataFrame([
            {"Annotation status": "Has gold evidence", "Claims": evidence_claims},
            {"Annotation status": "No gold evidence", "Claims": summary["records"] - evidence_claims},
        ])
        figure = px.pie(evidence_frame, names="Annotation status", values="Claims", hole=.48,
                        color="Annotation status", title="Claims with evidence annotations")
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
        st.plotly_chart(figure, use_container_width=True, key="dataset_evidence_coverage")

    st.subheader("Claim records")
    filter_col, search_col = st.columns([1, 2])
    available_labels = sorted({str(row["Label"]) for row in claim_table})
    selected_labels = filter_col.multiselect("Filter labels", available_labels, default=available_labels)
    search_text = search_col.text_input("Search claim text", key="dataset_claim_search")
    filtered = [
        row for row in claim_table
        if row["Label"] in selected_labels and search_text.casefold() in row["Claim"].casefold()
    ]
    st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matching claims")
    if filtered:
        st.dataframe(filtered[:500], hide_index=True, use_container_width=True)
        with st.expander("Inspect an individual claim"):
            selected_id = st.selectbox("Claim ID", [str(row["ID"]) for row in filtered], key="dataset_claim_detail")
            detail = next(row for row in claims if str(row.get("id")) == selected_id)
            st.markdown(f"**Claim:** {detail.get('claim', '')}")
            st.markdown(f"**Gold label:** {detail.get('label') or 'UNLABELED'}")
            evidence = detail.get("evidence") or []
            if evidence:
                st.markdown("**Gold evidence annotation:**")
                st.write(evidence)
            else:
                st.info("This record has no gold evidence annotation.")
    else:
        st.info("No claims match the current filters.")
else:
    st.info("No valid claims are available yet. Select a local JSONL file or upload a FEVER-format claims file to populate the charts.")
