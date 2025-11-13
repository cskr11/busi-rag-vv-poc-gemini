import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the environment variables from the .env file
load_dotenv()

# The ChatGoogleGenerativeAI class automatically looks for the GEMINI_API_KEY 
# or GOOGLE_API_KEY in the environment, so no explicit key passing is needed.

# 2. Initialize the Gemini Pro Chat Model
# Model Name: Use "gemini-pro" for the standard text model, 
# or "gemini-2.5-flash" or "gemini-2.5-pro" for the latest models.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", # Using the professional model for a powerful response
    temperature=0.7,        # A setting for creativity (0.0 is deterministic, 1.0 is highly creative)
)
print("✅ Gemini Pro LLM initialized.")

# 3. Define the LangChain Prompt and Chain
# Create a ChatPromptTemplate for a simple conversation
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative poet who writes in the style of Shakespeare."),
    ("human", "Write a short poem about {topic}."),
])

# Create a LangChain Expression Language (LCEL) chain
# The chain links the prompt input to the LLM model
chain = prompt | llm

# 4. Invoke the Chain
topic = "a computer"

try:
    print(f"\n--- Invoking Chain for topic: '{topic}' ---")
    response = chain.invoke({"topic": topic})

    print("\n--- Model Response ---")
    # response.content contains the generated text
    print(response.content)

except Exception as e:
    # This will catch the DefaultCredentialsError if the key wasn't loaded
    print(f"\n--- ❌ ERROR ---")
    print(f"Failed to connect to Gemini. Ensure GEMINI_API_KEY is set correctly.")
    print(f"Full Error: {e}")