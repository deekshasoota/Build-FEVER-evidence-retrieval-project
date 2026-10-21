import json
from pathlib import Path
import streamlit as st

st.title("Evaluation dashboard")
st.caption("Only computed run artifacts are shown. Sample fixtures are not FEVER benchmark results.")
results = Path("artifacts/results")
classification_path = results / "dev_metrics.json"
retrieval_path = results / "retrieval_metrics.json"
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

has_classification = classification_path.exists()
has_retrieval = retrieval_path.exists()
if not has_classification and not has_retrieval:
    st.info(
        "The dashboard is ready, but no evaluation reports have been generated for this deployment yet. "
        "Accuracy and its confusion matrix require a trained verifier and a labeled development split; "
        "retrieval metrics require FEVER evidence annotations. The small deployment preview cannot produce "
        "official benchmark scores on its own."
    )
    with st.expander("Generate the evaluation reports"):
        st.markdown("Download the official FEVER claim splits and Wikipedia sentence data, place them in `data/raw/`, and build the full index. Then run these commands from the project folder:")
        st.code(
            "python -m scripts.build_index data/raw/wiki-pages.jsonl --output artifacts/indexes/fever_index.json\n"
            "python -m scripts.train_model --train data/raw/train.jsonl --index artifacts/indexes/fever_index.json\n"
            "python -m scripts.run_evaluation --dev data/raw/dev.jsonl --index artifacts/indexes/fever_index.json\n"
            "python -m scripts.run_retrieval_evaluation --claims data/raw/dev.jsonl --index artifacts/indexes/fever_index.json --top-k 5",
            language="bash",
        )
        st.caption("The reports are written under `artifacts/results/`. Keep the full dataset and index local; only publish metrics you have actually computed.")

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
