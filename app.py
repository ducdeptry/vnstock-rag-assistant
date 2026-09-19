"""Streamlit demo UI for the Vnstock RAG assistant.

Run with: streamlit run app.py
"""
import streamlit as st

from src.rag import answer_question

st.set_page_config(page_title="Vnstock RAG Assistant", page_icon="📈")
st.title("📈 Vnstock RAG Assistant")
st.caption(
    "Ask questions about major Vietnamese companies (FPT, VNM, VIC, HPG, MWG, "
    "VCB, MSN, GAS, VHM, TCB) — answered using real data pulled via vnstock."
)

question = st.text_input("Ask a question", placeholder="What does FPT company do?")

if st.button("Ask") and question:
    with st.spinner("Retrieving context and generating an answer..."):
        result = answer_question(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Sources")
    for s in result["sources"]:
        st.write(f"- **{s['symbol']}** ({s['source']})")
