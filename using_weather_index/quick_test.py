import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the key
load_dotenv() 

# 2. Retrieve the key from the environment
# Use GOOGLE_API_KEY as the primary key. Fallback to GEMINI_API_KEY if needed.
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    # This check is still useful
    print("❌ ERROR: API key not loaded into environment variables. Check .env file.")
else:
    print("✅ API key loaded. Testing connection...")

# 3. Initialize and invoke LLM
try:
    # PASS THE KEY DIRECTLY TO THE CONSTRUCTOR
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.1, 
        google_api_key=api_key # <--- THE FIX
    )
    
    response = llm.invoke("What is the capital of France?")
    print("\n### LLM Test Success ###")
    print(response.content)
    
except Exception as e:
    print(f"\n❌ LLM Test Failed. Check your key's validity and permissions.")
    print(f"Error: {e}") 
    # The actual error message is vital here. Look for 'Unauthenticated' or 'Invalid credentials'.