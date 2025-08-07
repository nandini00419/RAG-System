# RAG-System
This project implements a Retrieval-Augmented Generation (RAG) pipeline to build an intelligent PDF-based Question Answering chatbot using LangChain, FAISS, and Groq’s LLMs like mixtral-8x7b-32768. The goal is to allow users to upload a PDF and ask questions about its content — with accurate, context-aware answers generated in real-time.


# RAG-System 📄🧠

This is a Retrieval-Augmented Generation (RAG) based PDF Q&A system using Groq's LLM, FAISS, and Streamlit.

### 📦 Features

- Upload any PDF (through backend code)
- Chunking & Embedding
- Groq LLM for answering
- Streamlit interface to interact

### 🔧 Installation

```bash
git clone https://github.com/nandini00419/RAG-System.git
cd RAG-System
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate (Windows)
pip install -r requirements.txt

