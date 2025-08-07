from langchain_groq import ChatGroq

def get_groq_llm():
    # Replace "your-groq-api-key-here" with your actual Groq API key
    api_key = "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"
    return ChatGroq(
        api_key=api_key,
        model_name="mixtral-8x7b-32768"
    )