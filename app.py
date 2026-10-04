"""NyayaAtlas - Streamlit app.

Run locally:  streamlit run app.py
On Hugging Face Spaces the index is built automatically on first start.
"""
import streamlit as st

from src import config
from src.rag import NOT_FOUND, answer

st.set_page_config(page_title="NyayaAtlas", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 980px; padding-top: 2rem;}
    .src-meta {color: #6b7280; font-size: 0.85rem; margin-bottom: .4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the document index (first start can take a few minutes)...")
def load_retriever():
    from src import ingest
    from src.retriever import Retriever

    if not config.CHUNKS_PATH.exists() or not config.CHROMA_DIR.exists():
        try:
            ingest.main()  # build the index from data/raw on first start
        except SystemExit as e:  # no documents / nothing extractable
            raise FileNotFoundError(str(e)) from e
    return Retriever()


st.title("⚖️ NyayaAtlas")
st.caption("Navigate Indian law and policy, with exact source citations.")
st.warning(
    "**Not legal advice.** NyayaAtlas is an information tool built on the documents "
    "indexed below. Laws change and answers can be incomplete or wrong. Verify with the "
    "official source or a qualified professional.",
    icon="⚠️",
)

with st.sidebar:
    st.header("About")
    st.write(
        "Answers come only from the indexed documents, and every answer shows the "
        "file and page it was taken from. If the documents don't contain the answer, "
        "NyayaAtlas says so instead of guessing."
    )
    st.markdown(
        "**How it works**\n"
        "1. Hybrid search (BM25 + embeddings)\n"
        "2. Results are fused and ranked\n"
        "3. The LLM answers only from those passages, with `[n]` citations\n"
        "4. Weak or missing evidence → *\"" + NOT_FOUND + "\"*"
    )
    mode = "LLM answers with citations" if config.ANTHROPIC_API_KEY else "Passages only (no API key set)"
    st.info(f"Mode: {mode}")

try:
    retriever = load_retriever()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pages_by_doc: dict[str, set] = {}
for c in retriever.chunks:
    pages_by_doc.setdefault(c["title"], set()).add(c["page"])
with st.expander(f"Indexed documents ({len(pages_by_doc)})"):
    for title, pages in sorted(pages_by_doc.items()):
        st.write(f"- **{title}**: {len(pages)} pages with text")

examples = [
    "What are the fundamental duties of a citizen of India?",
    "What minimum leverage ratio must AIFIs maintain?",
    "Is family pension taxed as salary income?",
]
st.write("Try an example:")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input("Ask a question", key="question", placeholder="e.g. What does Article 14 say about equality before the law?")

if question and question.strip():
    with st.spinner("Searching the documents and drafting a cited answer..."):
        result = answer(question, retriever)

    if result["abstained"]:
        st.warning(result["answer"], icon="🔍")
    else:
        st.subheader("Answer")
        with st.container(border=True):
            st.markdown(result["answer"])

    if result["sources"]:
        st.subheader("Sources")
        cited = set(result.get("cited", []))
        for i, s in enumerate(result["sources"], start=1):
            tag = " ✅ cited" if i in cited else ""
            with st.container(border=True):
                st.markdown(f"**[{i}] {s['title']}** · page {s['page']}{tag}")
                where = f"PDF page {s['page']}" if s["source"].lower().endswith(".pdf") else "text file (no pages)"
                st.markdown(
                    f"<div class='src-meta'>File: {s['source']} · {where}</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Show passage", expanded=i in cited):
                    st.text(s["text"])
