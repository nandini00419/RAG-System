import os
import streamlit as st
import pickle
from rag_pipeline import build_qa_pipeline

# Local Transformers setup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Paths
DATA_DIR = "data"
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

# ✅ Set Hugging Face Token
hf_token = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.markdown("### 💬 Chat with your document!")
st.markdown("Ask any question from your document below 👇")

# List PDFs in data directory
pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
selected_pdf = st.selectbox("📄 Choose a PDF:", pdf_files)

if selected_pdf:
    pdf_path = os.path.join(DATA_DIR, selected_pdf)
    vector_path = os.path.join(VECTOR_DIR, f"{selected_pdf}.pkl")

    # ✅ Load or build vectorstore
    if os.path.exists(vector_path):
        with open(vector_path, "rb") as f:
            vectordb = pickle.load(f)

        # ✅ Load local FLAN-T5 model
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        # ✅ Create a HuggingFace pipeline
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
        )

        # ✅ Wrap in LangChain-compatible model
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # ✅ Build QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    else:
        st.info("Building vector store for the document. Please wait...")
        qa_chain = build_qa_pipeline(pdf_path)
        vectordb = qa_chain.retriever.vectorstore
        with open(vector_path, "wb") as f:
            pickle.dump(vectordb, f)

    # ✅ User query input
    user_query = st.text_input("Ask a question:")
    if user_query:
        with st.spinner("🔍 Thinking..."):
            answer = qa_chain.run(user_query)
            st.success(answer)
