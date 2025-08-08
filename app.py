import os
import pickle
import streamlit as st
from rag_pipeline import build_qa_pipeline

DATA_DIR = "data"
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Streamlit UI Setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.markdown("### 💬 Chat with your document!")
st.markdown("Ask any question from your document below 👇")

# List PDFs
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
if not pdf_files:
    st.warning("No PDF files found in the data directory.")
    st.stop()

selected_pdf = st.selectbox("📄 Choose a PDF:", pdf_files)

if selected_pdf:
    pdf_path = os.path.join(DATA_DIR, selected_pdf)
    vector_path = os.path.join(VECTOR_DIR, f"{selected_pdf}.pkl")

    @st.cache_resource(show_spinner=False)
    def load_qa_chain(pdf_path, vector_path):
        # Try loading saved vectorstore for speed
        if os.path.exists(vector_path):
            try:
                with open(vector_path, "rb") as f:
                    vectordb = pickle.load(f)
                # Rebuild QA chain with existing vectorstore and fresh LLM
                from utils.groq_llm import get_groq_llm
                from langchain.chains import RetrievalQA
                llm = get_groq_llm()
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
                return qa_chain, vectordb
            except Exception as e:
                st.warning(f"Vectorstore could not be loaded: {str(e)}. Rebuilding.")

        # Build new QA pipeline, store its vectorstore for future use
        qa_chain = build_qa_pipeline(pdf_path)
        vectordb = qa_chain.retriever.vectorstore
        with open(vector_path, "wb") as f:
            pickle.dump(vectordb, f)
        return qa_chain, vectordb

    qa_chain, vectordb = load_qa_chain(pdf_path, vector_path)

    # Chat history
    chat_key = f"chat_history_{selected_pdf}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    user_query = st.text_input("Ask a question:")

    if user_query:
        with st.spinner("🔍 Thinking..."):
            try:
                result = qa_chain({"query": user_query}, return_only_outputs=True)
                answer = result["result"]
                st.session_state[chat_key].append(("You", user_query))
                st.session_state[chat_key].append(("Bot", answer))
                st.success(answer)
                # Optionally show retrieved source docs
                if "source_documents" in result:
                    with st.expander("Show source"):
                        for doc in result["source_documents"]:
                            st.markdown(f"- {doc.metadata.get('source', 'source unknown')} (p.{doc.metadata.get('page', '?')}): {doc.page_content[:400]}...")
            except Exception as e:
                st.error(f"Error: {e}")

    # Render chat history
    st.markdown("#### 🗣️ Conversation")
    for speaker, message in st.session_state[chat_key]:
        color = "#007aff" if speaker == "You" else "#222"
        st.markdown(f"<div style='color:{color}'><b>{speaker}:</b> {message}</div>", unsafe_allow_html=True)

