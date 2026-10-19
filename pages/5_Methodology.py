from pathlib import Path
import streamlit as st

st.title("Methodology and About")
for name in ("methodology.md", "limitations.md"):
    path=Path("docs")/name
    with st.expander(name.replace(".md", "").replace("_", " ").title(), expanded=name == "methodology.md"):
        st.markdown(path.read_text(encoding="utf-8"))
st.markdown("Dataset and paper: [official FEVER dataset](https://fever.ai/dataset/fever.html) · [original FEVER paper](https://aclanthology.org/N18-1074/)")
