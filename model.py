import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load environment variables from .env file
load_dotenv()

# 2. Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# 3. Initialize Groq client
client = Groq(api_key=api_key)

# 4. Create a chat completion request
chat_completion = client.chat.completions.create(
    model="llama3-8b-8192",  # You can change to another model from Groq's list
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello from my RAG-System project!"}
    ],
    temperature=0.7,
    max_tokens=512
)

# 5. Print the AI's response
print("🤖 AI says:", chat_completion.choices[0].message.content)
