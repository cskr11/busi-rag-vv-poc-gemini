import os
from pydantic import SecretStr
# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
# EMBEDDINGS (Local HuggingFace)
from langchain_community.embeddings import HuggingFaceEmbeddings
# VECTOR STORE (Chroma - used for stability)
from langchain_community.vectorstores import Chroma
# Core LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# FIX: Import StrOutputParser to force the output into a simple string
from langchain_core.output_parsers import StrOutputParser

# Import modular components (assuming data_prep.py is in the same directory)
from weather_data_working.data_prep_localfile import create_and_chunk_documents
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Chroma Configuration ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "weather_rag_collection"

# --- Authentication Fix (For the Gemini LLM) ---
api_key_str = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key_str:
    raise ValueError("API Key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")

# Wrap the API key securely
api_key = SecretStr(api_key_str)
# --- End Auth Fix ---

# --- RAG Core Logic using LCEL pipe operator (|) ---

def create_rag_chain(llm, retriever):
    """
    Creates the Retrieval-Augmented Generation chain using LangChain Expression Language (LCEL).

    Args:
        llm: The initialized LangChain LLM object (ChatGoogleGenerativeAI).
        retriever: The Chroma vector store retriever object.

    Returns:
        The complete RAG chain runnable object.
    """
    system_prompt = (
        "You are an expert risk assessment analyst. Answer the user's question based "
        "only on the provided context (weather data). "
        "If the context does not contain the answer, politely state that you cannot provide an answer. "
        "\n\nCONTEXT: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Function to format retrieved documents into a single string for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Instantiate the output parser
    output_parser = StrOutputParser()

    # The RAG chain using LCEL
    rag_chain = (
        # 1. Input Mapping: Maps the user 'input' to the retriever/context pipeline
        # and passes the user 'input' through to the prompt.
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        # 2. Prompt application: Fills the template with context and input
        | prompt
        # 3. LLM call: Sends the completed prompt to Gemini
        | llm
        # 4. FIX: Ensures the output is a simple string, resolving the 'dict' error
        | output_parser
    )
    return rag_chain

def run_rag_query(query: str):
    """Runs the full RAG pipeline."""

    # --- 0. Initialize Embeddings Client (HUGGINGFACE) ---
    print("\nInitializing Local HuggingFace Embeddings Client...")
    # Using a fast, local model for embedding chunks
    embeddings_client = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # --- 1. Connect to Vector Store and Index Check (CHROMA) ---
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings_client,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # Check if the collection is empty. If so, index the data.
    if not vector_store._collection.count():
        print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' is empty. Indexing data now...")
        # Load and chunk data from OpenSearch
        chunks = create_and_chunk_documents()

        vector_store.add_documents(chunks)
        print("Indexing complete and persisted to disk.")
    else:
        print(f"Chroma collection '{CHROMA_COLLECTION_NAME}' found with {vector_store._collection.count()} items. Proceeding with retrieval.")

    # Define the Retriever (top 2 closest chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # --- 2. Initialize LLM (Gemini 2.5 Pro) ---
    print("\nInitializing Gemini 2.5 Pro LLM and RAG chain (Explicit Auth)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.1,
        google_api_key=api_key,
        transport="rest"
    )

    qa_chain = create_rag_chain(llm, retriever)

    # --- 3. Execute Query ---
    print(f"\n--- Running RAG Query: {query} ---")

    # The chain now accepts the string directly and returns a simple string 'result'
    result = qa_chain.invoke(query)

    # --- 4. Output Results and Sources ---
    print("\n### RAG Final Answer (Gemini 2.5 Pro) ###")
    # Print the simple string result
    print(result)

    print("\n### Source Documents (Context Retrieved from Chroma) ###")

    # Use the retriever separately to get the source documents for display
    source_documents = retriever.invoke(query)

    if source_documents:
        for i, source_doc in enumerate(source_documents):
            print(f"\n--- Source Document {i+1} ---")
            print(f"City: {source_doc.metadata.get('city', 'N/A')}")
            print(f"Condition: {source_doc.metadata.get('condition', 'N/A')}")
            print(f"Content Used: {source_doc.page_content}")
    else:
        print("No source documents were retrieved for this query.")

    return result

if __name__ == "__main__":
    try:
        query_to_run = "Which city is under a severe weather alert, and what are the details about wind speed and flights?"
        run_rag_query(query_to_run)
    except Exception as e:
        print(f"\n❌ A critical error occurred during the RAG pipeline execution.")
        print(f"Error details: {e}")