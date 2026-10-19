"""Evaluate saved verifier on a labeled development JSONL split."""
import argparse, json, pickle
from pathlib import Path
from src.data.load_data import load_claims
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search
from src.verification.feature_extractor import extract_features
from src.evaluation.verification_metrics import classification_metrics
from src.evaluation.fever_evaluation import acceptable_evidence_sets, fever_score
from src.evaluation.retrieval_metrics import evidence_set_complete

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dev",required=True); p.add_argument("--index",required=True)
    p.add_argument("--model",default="artifacts/models/verifier.pkl"); p.add_argument("--limit",type=int)
    p.add_argument("--output",default="artifacts/results/dev_metrics.json"); a=p.parse_args()
    rows=load_claims(a.dev); rows=rows[:a.limit] if a.limit else rows
    model=pickle.loads(Path(a.model).read_bytes()); index=InvertedIndex.load(a.index)
    gold=[]; predicted=[]; evidence_ok=[]
    for row in rows:
        if row.get("label") not in {"SUPPORTS","REFUTES","NOT ENOUGH INFO"}: continue
        evidence=search(row["claim"],index,10)
        gold.append(row["label"]); predicted.append(str(model.predict([extract_features(row["claim"],evidence)])[0]))
        groups=acceptable_evidence_sets(row.get("evidence"))
        evidence_ok.append(row["label"] == "NOT ENOUGH INFO" or evidence_set_complete({str(item["id"]) for item in evidence}, groups))
    report=classification_metrics(gold,predicted)
    report["fever_score"] = fever_score(gold, predicted, evidence_ok)
    report.update({"split":"development","records":len(gold),"retrieval":"BM25","model":str(a.model)})
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
