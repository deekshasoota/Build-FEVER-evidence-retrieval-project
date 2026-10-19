from pathlib import Path
import streamlit as st
from src.data.load_data import load_claims
from src.data.validate_data import summarize_claims

st.title("Dataset Explorer")
st.caption("Gold labels and annotations belong to the dataset; they are not model predictions.")
path = st.text_input("Claims JSONL path", "data/raw/dev.jsonl")
if Path(path).exists():
    claims = load_claims(path)
    summary = summarize_claims(claims)
    st.metric("Valid claims loaded", summary["records"])
    st.write("Gold label distribution", summary["labels"])
    st.write("Claims with evidence annotations", summary["with_evidence"])
    st.dataframe([{k:r.get(k) for k in ("id","claim","label","evidence")} for r in claims[:100]], use_container_width=True)
else:
    st.info("Dataset not prepared. Place an official FEVER claims JSONL file under data/raw/ and enter its path.")
