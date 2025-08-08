from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

def get_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file.")

    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192"  # Updated supported model
    )
