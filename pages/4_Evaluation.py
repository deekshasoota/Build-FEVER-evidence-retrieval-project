import json
import gzip
from pathlib import Path
import sqlite3
import tempfile
import plotly.express as px
import pandas as pd
import streamlit as st

from src.data.load_data import load_claims
from src.data.validate_data import summarize_claims
from src.indexing.sqlite_index import SQLiteEvidenceIndex
from src.retrieval import bm25_retriever, tfidf_retriever

st.title("Evaluation dashboard")
st.caption("Only computed run artifacts are shown. Sample fixtures are not FEVER benchmark results.")
results = Path("artifacts/results")
classification_path = results / "dev_metrics.json"
retrieval_path = results / "retrieval_metrics.json"
comparison_path = results / "retriever_comparison.json"
preview_archive = Path("data/demo/fever_demo.sqlite3.gz")
preview_database = Path("data/demo/fever_demo.sqlite3")

@st.cache_resource(show_spinner="Preparing the preview corpus for this comparison…")
def load_comparison_resources(index_path: str, modified_ns: int, compressed: bool):
    if compressed:
        database_path = Path(tempfile.gettempdir()) / "fever_dashboard_preview.sqlite3"
        if not database_path.exists() or database_path.stat().st_mtime_ns < modified_ns:
            with gzip.open(index_path, "rb") as source, database_path.open("wb") as target:
                target.write(source.read())
    else:
        database_path = Path(index_path)
    index = SQLiteEvidenceIndex(database_path)
    if index.document_count > 100_000:
        raise RuntimeError("The interactive comparison needs an index of 100,000 sentences or fewer. Use the included preview corpus.")
    if hasattr(index, "all_documents"):
        records = index.all_documents()
    else:
        # Older local Streamlit sessions may still hold the pre-pagination class.
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=30) as db:
            rows = db.execute("SELECT doc_id, page, sentence_id, text FROM evidence_fts ORDER BY doc_id").fetchall()
        records = [
            {"id": row[0], "page": row[1], "sentence_id": row[2], "text": row[3]}
            for row in rows
        ]
    preprocessing = getattr(index, "preprocessing", {})
    vectorizer, tfidf_matrix = tfidf_retriever.build_index(records, preprocessing)
    return index, records, vectorizer, tfidf_matrix

def ranking_overlap(rankings: dict[str, list[dict]], top_k: int) -> pd.DataFrame:
    names = list(rankings)
    sets = {name: {str(row["id"]) for row in rows[:top_k]} for name, rows in rankings.items()}
    return pd.DataFrame(
        [[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) if sets[a] | sets[b] else 0.0 for b in names] for a in names],
        index=names, columns=names,
    )

st.markdown("Upload saved evaluation reports to display their results here.")
upload_col1, upload_col2, upload_col3 = st.columns(3)
comparison_upload = upload_col1.file_uploader("Retrieval comparison", type="json", key="comparison_report_upload")
classification_upload = upload_col2.file_uploader("Fact verification", type="json", key="classification_report_upload")
retrieval_upload = upload_col3.file_uploader("Single retriever run", type="json", key="retrieval_report_upload")

def load_report(path: Path, uploaded_file):
    try:
        if uploaded_file is not None:
            report = json.loads(uploaded_file.getvalue())
        elif path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            return None
        if not isinstance(report, dict):
            st.error(f"{uploaded_file.name if uploaded_file else path.name} must contain a JSON object report.")
            return None
        return report
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f"Could not read {uploaded_file.name if uploaded_file else path.name}: {exc}")
    return None

comparison_report = load_report(comparison_path, comparison_upload)
classification_report_data = load_report(classification_path, classification_upload)
retrieval_report = load_report(retrieval_path, retrieval_upload)
full_index_path = Path("artifacts/indexes/fever_index.meta.json")
preview_index_path = Path("data/demo/fever_demo.meta.json")
index_path = full_index_path if full_index_path.exists() else preview_index_path

if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    st.subheader("Index build" if index_path == full_index_path else "Deployment preview index")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentences", f"{index.get('sentences', 0):,}")
    c2.metric("Terms", f"{index['terms']:,}" if isinstance(index.get("terms"), int) else "—")
    c3.metric("Build time", f"{index['build_seconds']:.2f} s" if isinstance(index.get("build_seconds"), (int, float)) else "—")
    c4.metric("Index size", f"{index.get('index_bytes', 0) / (1024 * 1024):.2f} MB")
    if index.get("deployment_demo"):
        st.caption("This is a sampled deployment preview. Use the full local index for academic evaluation.")

has_classification = classification_report_data is not None
has_retrieval = retrieval_report is not None
has_comparison = comparison_report is not None
if not has_comparison:
    st.subheader("Required retrieval comparison matrix")
    st.caption("Scores stay blank until the development claims with gold evidence annotations are evaluated.")
    pending_models = ["BM25", "TF-IDF", "Sentence Embedding", "Hybrid RRF"]
    st.dataframe([
        {"Ranking method": model, "Precision@10": None, "Recall@10": None,
         "MAP@10": None, "nDCG@10": None, "Status": "Awaiting labeled benchmark run"}
        for model in pending_models
    ], hide_index=True, use_container_width=True)
    st.markdown("**Preprocessing in the index builder:** Unicode normalization, case folding, tokenization, optional stop-word removal (negation retained), and optional Porter stemming. The same saved settings are applied to BM25, Boolean, and TF-IDF queries.")
if not has_classification:
    st.subheader("Fact-verification confusion matrix")
    st.caption("Awaiting a trained verifier and labeled development predictions; no classification counts are available yet.")
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    st.dataframe([
        {"Gold \\ Predicted": label, **{f"{predicted} · predicted": "—" for predicted in labels}}
        for label in labels
    ], hide_index=True, use_container_width=True)
if not has_classification and not has_retrieval and not has_comparison:
    st.info(
        "No benchmark run has been saved to this deployment yet. The interactive comparison below will show "
        "Precision@10, Recall@10, MAP@10, nDCG@10, and precision–recall curves for the four required rankers "
        "once the labeled evaluation data is available. The preview corpus alone cannot produce valid scores."
    )
    st.info("After a labeled benchmark run, upload its retrieval comparison report above. The matrix and precision–recall chart will populate from that report. No scores are calculated from the deployment preview.")
    sample_claims_path = Path("data/sample/claims.jsonl")
    if sample_claims_path.exists():
        sample_summary = summarize_claims(load_claims(sample_claims_path))
        st.subheader("Available dataset preview")
        st.caption("This chart summarizes the tiny bundled example only; it is not a retrieval performance comparison.")
        preview_labels = pd.DataFrame([
            {"Gold label": label, "Claims": count}
            for label, count in sorted(sample_summary["labels"].items())
        ])
        preview_cols = st.columns([1, 2])
        preview_cols[0].metric("Example claims", f"{sample_summary['records']:,}")
        preview_cols[0].metric("With evidence annotations", f"{sample_summary['with_evidence']:,}")
        if not preview_labels.empty:
            preview_figure = px.bar(
                preview_labels, x="Gold label", y="Claims", color="Gold label", text="Claims",
                title="Example claims by gold label",
            )
            preview_figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
            preview_cols[1].plotly_chart(preview_figure, use_container_width=True, key="preview_claim_labels")

if has_comparison:
    report = comparison_report
    st.subheader("Ranking model comparison")
    index_size = report.get("evidence_index_documents")
    index_size_label = f"{index_size:,}" if isinstance(index_size, int) else "unknown"
    st.caption(
        f"Split: {report.get('split', 'unknown')} · Claims with evidence: "
        f"{report.get('claims_with_evidence', 0):,} · Evidence index: "
        f"{index_size_label} sentences · "
        f"Cutoff: @{report.get('top_k', 10)}"
    )
    st.caption(report.get("evidence_index_scope", ""))
    domains = report.get("by_domain", {})
    domain_names = sorted(name for name in domains if name != "Unassigned")
    domain_count = len(domain_names)
    st.metric("Labeled domains found", f"{domain_count} / 21")
    if report.get("claims_missing_domain", 0):
        st.warning(
            f"{report['claims_missing_domain']:,} evaluated claims have no `{report.get('domain_field', 'domain')}` label. "
            "They are included in overall scores but cannot appear in a domain-specific row."
        )
    if domain_count != 21:
        st.info("Domain-level results reflect only labeled domains present in the input file. The project needs all 21 domain labels for the complete comparison.")

    domain_options = ["Overall", *domain_names]
    selected_domain = st.selectbox("View results for", domain_options, key="evaluation_domain")
    selected = report.get("overall", {}) if selected_domain == "Overall" else domains.get(selected_domain, {})
    metric_rows = []
    chart_rows = []
    metric_chart_rows = []
    metric_chart_labels = {
        "precision_at_10": "Precision@10",
        "recall_at_10": "Recall@10",
        "map_at_10": "MAP@10",
        "ndcg_at_10": "nDCG@10",
    }
    for model_name, summary in selected.items():
        metrics = summary.get("metrics", {})
        metric_rows.append({
            "Ranking method": model_name,
            "Claims": summary.get("claims_with_evidence", 0),
            "Precision@10": metrics.get("precision_at_10"),
            "Recall@10": metrics.get("recall_at_10"),
            "MAP@10": metrics.get("map_at_10"),
            "nDCG@10": metrics.get("ndcg_at_10"),
            "Mean latency (ms)": summary.get("mean_latency_ms"),
        })
        for metric_key, metric_label in metric_chart_labels.items():
            value = metrics.get(metric_key)
            if isinstance(value, (int, float)):
                metric_chart_rows.append({"Ranking method": model_name, "Metric": metric_label, "Score": value})
        for point in summary.get("curve", []):
            chart_rows.append({
                "Ranking method": model_name,
                "Precision": point.get("precision"),
                "Recall": point.get("recall"),
                "Cutoff": point.get("k"),
            })
    if metric_rows:
        st.dataframe(metric_rows, hide_index=True, use_container_width=True)
    if metric_chart_rows:
        metric_figure = px.bar(
            metric_chart_rows, x="Ranking method", y="Score", color="Metric", barmode="group",
            title=f"Retrieval metric comparison · {selected_domain}",
            category_orders={"Metric": list(metric_chart_labels.values())},
        )
        metric_figure.update_yaxes(range=[0, 1], title="Score (0–1)")
        metric_figure.update_layout(legend_title_text="Metric", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(metric_figure, use_container_width=True, key="retrieval_metric_comparison")
    if chart_rows:
        figure = px.line(
            chart_rows, x="Recall", y="Precision", color="Ranking method",
            markers=True, hover_data=["Cutoff"],
            title=f"Precision–recall curve · {selected_domain}",
        )
        figure.update_layout(legend_title_text="Ranking method", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, use_container_width=True, key="precision_recall_curve")

    if domains:
        with st.expander("All domains · full results table"):
            domain_rows = []
            for domain_name, models in sorted(domains.items()):
                for model_name, summary in models.items():
                    metrics = summary.get("metrics", {})
                    domain_rows.append({
                        "Domain": domain_name,
                        "Ranking method": model_name,
                        "Claims": summary.get("claims_with_evidence", 0),
                        "Precision@10": metrics.get("precision_at_10"),
                        "Recall@10": metrics.get("recall_at_10"),
                        "MAP@10": metrics.get("map_at_10"),
                        "nDCG@10": metrics.get("ndcg_at_10"),
                    })
            if domain_rows:
                st.dataframe(domain_rows, hide_index=True, use_container_width=True)
    st.caption(report.get("note", ""))

st.subheader("Quick retrieval comparison")
st.caption("Try a claim or topic to see how the methods rank evidence. The heatmap measures top-result overlap, not accuracy; use gold-labeled evaluation reports for performance scores.")
comparison_index_path = (
    preview_database if preview_database.exists()
    else preview_archive if preview_archive.exists()
    else Path("artifacts/indexes/fever_index.sqlite3")
)
if comparison_index_path.exists():
    with st.form("quick_retrieval_comparison"):
        query_col, k_col = st.columns([4, 1])
        comparison_query = query_col.text_input("Claim or topic", "The Pacific Ocean is the largest ocean on Earth.", key="live_comparison_query")
        comparison_k = k_col.number_input("Top results", min_value=3, max_value=10, value=5, step=1, key="live_comparison_top_k")
        compare_submitted = st.form_submit_button("Compare methods", type="primary")
    if compare_submitted:
        if not comparison_query.strip():
            st.warning("Enter a claim or topic first.")
        else:
            try:
                with st.spinner("Ranking evidence with BM25 and TF-IDF…"):
                    idx, records, vectorizer, matrix = load_comparison_resources(
                        str(comparison_index_path), comparison_index_path.stat().st_mtime_ns,
                        comparison_index_path.suffix == ".gz",
                    )
                    bm25_rows = bm25_retriever.search(comparison_query, idx, int(comparison_k))
                    tfidf_rows = tfidf_retriever.search_index(
                        comparison_query, records, vectorizer, matrix, int(comparison_k),
                        getattr(idx, "preprocessing", {}),
                    )
                    rankings = {
                        "BM25": bm25_rows,
                        "TF-IDF": tfidf_rows,
                    }
                overlap = ranking_overlap(rankings, int(comparison_k))
                overlap.index.name = "Retriever"
                overlap.columns.name = "Retriever"
                overlap_figure = px.imshow(
                    overlap, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale="Tealgrn",
                    labels={"x": "Retriever", "y": "Retriever", "color": "Jaccard overlap"},
                    title=f"Top-{int(comparison_k)} result overlap",
                )
                overlap_figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(overlap_figure, use_container_width=True, key="live_retriever_overlap")
                result_tabs = st.tabs(list(rankings))
                for tab, (method, rows) in zip(result_tabs, rankings.items()):
                    with tab:
                        if rows:
                            st.dataframe([
                                {"Rank": row.get("rank"), "Page": str(row.get("page", "")).replace("_", " "),
                                 "Sentence": row.get("text", "")}
                                for row in rows
                            ], hide_index=True, use_container_width=True)
                        else:
                            st.info(f"{method} returned no results for this query.")
            except Exception as exc:
                st.error(f"The live comparison could not run: {exc}")
else:
    st.info("No small evidence preview index is available for the live comparison. Upload a labeled comparison report above to view its benchmark charts.")

st.caption("The quick comparison uses BM25 and TF-IDF so results load without downloading or encoding an embedding model. Sentence Embedding and Hybrid RRF scores appear in the labeled benchmark report when you run the full evaluation.")

if has_classification:
    report = classification_report_data
    st.subheader("Fact verification")
    st.caption(f"Split: {report.get('split', 'unknown')} · Claims: {report.get('records', 'unknown')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{report.get('accuracy', 0):.3f}")
    c2.metric("FEVER score", f"{report.get('fever_score', 0):.3f}")
    macro_f1 = report.get("classification_report", {}).get("macro avg", {}).get("f1-score")
    c3.metric("Macro F1", f"{macro_f1:.3f}" if macro_f1 is not None else "—")
    labels = report.get("labels", [])
    matrix = report.get("confusion_matrix", [])
    if labels and matrix:
        st.write("Confusion matrix · rows are gold labels; columns are predictions")
        st.dataframe({"Gold label": labels, **{f"Predicted {label}": [matrix[r][c] for r in range(len(labels))] for c, label in enumerate(labels)}}, hide_index=True, use_container_width=True)
    st.write("Per-class report")
    per_class = {label: report.get("classification_report", {}).get(label, {}) for label in labels}
    if per_class:
        st.dataframe([{"Label": label, **metrics} for label, metrics in per_class.items()], hide_index=True, use_container_width=True)

if has_retrieval:
    report = retrieval_report
    metrics = report.get("metrics", {})
    st.subheader("Evidence retrieval")
    st.caption(f"Retriever: {report.get('retriever', 'unknown')} · Claims with annotated evidence: {report.get('claims_with_evidence', 0)} · K={report.get('top_k', '?')}")
    cols = st.columns(5)
    names = [("precision_at_k", "Precision@K"), ("recall_at_k", "Recall@K"), ("mrr", "MRR"), ("ndcg_at_k", "nDCG@K"), ("evidence_set_complete", "Evidence complete")]
    for col, (key, label) in zip(cols, names):
        col.metric(label, f"{metrics.get(key, 0):.3f}")
    st.metric("Mean retrieval latency", f"{report.get('mean_latency_ms', 0):.2f} ms / claim")
    if report.get("note"):
        st.caption(report["note"])
