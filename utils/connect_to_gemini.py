import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr # <-- Import SecretStr for the real fix

# Load environment variables from .env file
load_dotenv()

# --- START OF FIX ---

# 1. Load the key from the environment
api_key_str = os.getenv("GOOGLE_API_KEY") or ""
if not api_key_str:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file.")
else:
    print("✅ GOOGLE_API_KEY loaded.")

# 2. Convert to SecretStr for the constructor
api_key = SecretStr(api_key_str)

# 3. Initialize the LLM, passing the key AND the transport
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=api_key, # <-- Explicitly pass the key
    transport="rest"        # <-- Force the REST (API Key) transport
)
# --- END OF FIX ---


# Example LangChain pipeline
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
chain = prompt | llm

# Invoke the chain
try:
    response = chain.invoke({"topic": "LangChain"})
    print("\n--- SUCCESS ---")
    print(response.content)
except Exception as e:
    print(f"\n--- ❌ TEST FAILED ---")
    print(f"Error: {e}")