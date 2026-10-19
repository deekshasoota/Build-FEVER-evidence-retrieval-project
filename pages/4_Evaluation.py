import json
from pathlib import Path
import streamlit as st

st.title("Evaluation dashboard")
st.caption("Only computed run artifacts are shown. Sample fixtures are not FEVER benchmark results.")
results = Path("artifacts/results")
classification_path = results / "dev_metrics.json"
retrieval_path = results / "retrieval_metrics.json"
index_path = Path("artifacts/indexes/fever_index.meta.json")

if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    st.subheader("Index build")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentences", f"{index.get('sentences', 0):,}")
    c2.metric("Terms", f"{index.get('terms', 0):,}")
    c3.metric("Build time", f"{index.get('build_seconds', 0):.2f} s")
    c4.metric("Index size", f"{index.get('index_bytes', 0) / (1024 * 1024):.2f} MB")

has_classification = classification_path.exists()
has_retrieval = retrieval_path.exists()
if not has_classification and not has_retrieval:
    st.info("No evaluation runs yet. Use the train, development evaluation, and retrieval evaluation commands in the README. This page will not invent placeholder scores.")

if has_classification:
    report = json.loads(classification_path.read_text(encoding="utf-8"))
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
    report = json.loads(retrieval_path.read_text(encoding="utf-8"))
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

with st.expander("Raw run details"):
    for path in (classification_path, retrieval_path, index_path):
        if path.exists():
            st.markdown(f"**{path.name}**")
            st.json(json.loads(path.read_text(encoding="utf-8")))
