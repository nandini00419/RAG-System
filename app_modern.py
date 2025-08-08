import os
import pickle
import streamlit as st
from rag_pipeline import build_qa_pipeline

DATA_DIR = "data"
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)


def set_theme():
    st.markdown("""
        <style>
        .main { background-color: #f4f8fc; }
        .stApp { background-color: #f4f8fc; }
        .css-1dp5vir, .css-1v0mbdj { background: #fff!important; }
        </style>
    """, unsafe_allow_html=True)


set_theme()

# ---- Sidebar ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=54)
    st.header("Settings")
    st.markdown("Adjust PDF, model & chatting options below!")
    # PDF selector
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    selected_pdf = st.selectbox("📄 Choose PDF:", pdf_files)

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 10)
        show_sources = st.checkbox("Show Answer Sources", value=True)

    st.divider()
    st.caption("Powered by FAISS, LangChain & Groq LLM / FLAN-T5")

# -------- Page Header --------
st.markdown(
    "<h1 style='color:#007aff;font-size:36px;font-family:sans-serif;'>💬 RAG PDF Chatbot</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#222;font-size:18px;font-family:sans-serif;'>Upload your document, ask anything, and see context-aware answers powered by state-of-the-art AI!</p>",
    unsafe_allow_html=True
)

if not selected_pdf:
    st.info("Please upload or select a PDF from the sidebar.")
    st.stop()

pdf_path = os.path.join(DATA_DIR, selected_pdf)
vector_path = os.path.join(VECTOR_DIR, f"{selected_pdf}.pkl")


# ---- Load or Build QA ----
@st.cache_resource(show_spinner=True)
def load_qa_chain(pdf_path, vector_path, chunk_size, chunk_overlap):
    # Try to load vectorstore first
    if os.path.exists(vector_path):
        try:
            with open(vector_path, "rb") as f:
                vectordb = pickle.load(f)
            # Rebuild QA chain with current LLM
            from utils.groq_llm import get_groq_llm
            from langchain.chains import RetrievalQA
            llm = get_groq_llm()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
            return qa_chain, vectordb
        except Exception as e:
            st.warning(f"Failed to load vectorstore: {str(e)}. Rebuilding.")

    # New build using rag_pipeline with options
    qa_chain = build_qa_pipeline(
        pdf_path)  # You can modify this to pass chunk_size, chunk_overlap if your pipeline supports
    vectordb = qa_chain.retriever.vectorstore
    with open(vector_path, "wb") as f:
        pickle.dump(vectordb, f)
    return qa_chain, vectordb


qa_chain, vectordb = load_qa_chain(pdf_path, vector_path, chunk_size, chunk_overlap)

# ---- Session State for Chat ----
chat_key = f"chat_history_{selected_pdf}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []

# -------- Chat UI --------
st.markdown("## 🗣️ Your Conversation")


def render_chat():
    for speaker, message in st.session_state[chat_key]:
        if speaker == "You":
            st.markdown(
                f"""
                <div style='background:#eaf4fe;padding:10px 18px;border-radius:15px;margin:6px 0;display:inline-block;max-width:80%;'>
                  <b>🧑 You:</b> {message}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style='background:#f5f5f7;padding:10px 18px;border-radius:15px;margin:6px 0;display:inline-block;max-width:80%;'>
                  <b>🤖 Bot:</b> {message}
                </div>
                """, unsafe_allow_html=True)


render_chat()

user_query = st.text_input("💬 Type your question and press Enter...", key=selected_pdf)

if user_query:
    with st.spinner("🤖 Fetching your answer..."):
        try:
            result = qa_chain({"query": user_query}, return_only_outputs=True)
            answer = result["result"]
            st.session_state[chat_key].append(("You", user_query))
            st.session_state[chat_key].append(("Bot", answer))
            st.rerun()  # Force re-render of chat bubbles
        except Exception as e:
            st.session_state[chat_key].append(("Bot", f"Error: {e}"))
            st.error(f"Error: {e}")

# -------- Show sources if set in sidebar --------
if show_sources and user_query and "result" in locals() and "source_documents" in result:
    with st.expander("📖 Sources used for this answer"):
        for i, doc in enumerate(result["source_documents"]):
            s = doc.metadata.get('source', 'Unknown source')
            pg = doc.metadata.get('page', '?')
            st.markdown(
                f"""
                **Source {i + 1}:** {s}, page {pg}
                ```
                {doc.page_content[:750]}...
                ```
                """,
                unsafe_allow_html=True,
            )


# -------- Chat Download Option --------
def export_chat(chat_history):
    return "\n".join([f"{s}: {m}" for s, m in chat_history])


with st.sidebar:
    st.subheader("💾 Export Chat")
    chat_txt = export_chat(st.session_state[chat_key])
    st.download_button("Download as TXT", chat_txt, file_name=f"chat_{selected_pdf}.txt")
    if st.button("🗑️ Clear Conversation"):
        st.session_state[chat_key] = []
        st.experimental_rerun()

st.markdown("---")
st.caption("Demo RAG PDF Chatbot — UI enhanced by Streamlit. Your feedback is welcome! 🚀")
