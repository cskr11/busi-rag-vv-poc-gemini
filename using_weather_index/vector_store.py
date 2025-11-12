import os
import sys
import time  # <-- Import time for a small delay
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from typing import Any, Dict, List # <-- Added List for typing
from langchain_core.documents import Document # <-- Added Document for typing

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- Configuration (Centralized or using Environment Variables) ---
OPENSEARCH_URL = "http://localhost:9200"
OPENSEARCH_VECTOR_INDEX = "weather-vector-store"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # Use the correct model name
EMBEDDING_DIMENSION = 768
VECTOR_FIELD = "vector_field"


def get_opensearch_vector_store():
    """Initializes the embedding model and OpenSearch Vector Store connection."""

    # --- START OF FIX ---
    # 1. Retrieve the API Key from the environment
    api_key_str = os.getenv("GOOGLE_API_KEY")

    # 2. Check if the key was loaded successfully
    if not api_key_str:
        print(
            "❌ ERROR: GOOGLE_API_KEY environment variable not found. Check your .env file."
        )
        sys.exit(1)

    # 3. Convert the string to a SecretStr for Pydantic/LangChain
    api_key = SecretStr(api_key_str)
    # --- END OF FIX ---

    print("\nInitializing Google Generative AI Embeddings...")

    # 4. Pass the key EXPLICITLY and set transport="rest"
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=api_key,
        transport="rest"  # <-- THIS IS THE CRITICAL FIX
    )

    print(
        f"Connecting to OpenSearch at {OPENSEARCH_URL} and index '{OPENSEARCH_VECTOR_INDEX}'..."
    )

    # Initialize the vector store connection
    vector_store = OpenSearchVectorSearch(
        index_name=OPENSEARCH_VECTOR_INDEX,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        vector_field=VECTOR_FIELD,
        text_field="text",
        # k-NN index configuration
        engine="nmslib",
        space_type="l2",
        dim=EMBEDDING_DIMENSION,
        m=16,
        ef_construction=200,
    )
    return vector_store


def index_documents(vector_store: OpenSearchVectorSearch, documents: List[Document]):
    """Adds a list of LangChain Documents (chunks) to the OpenSearch index."""
    print("Indexing chunks in batch...")
    try:
        ids = vector_store.add_documents(documents)
        print(
            f"Batch indexing complete. Total documents/chunks indexed: {len(ids)}"
        )
        return ids
    except Exception as e:
        print(f"❌ ERROR during indexing: {e}")
        # This is where the 'DESCRIPTOR' error would have happened, but now it's auth.
        # This will now be fixed.
        raise e


# delete index function
def delete_index(vector_store: OpenSearchVectorSearch, index_name: str):
    """Deletes the specified OpenSearch index if it exists."""
    print(f"Attempting to delete index '{index_name}'...")
    try:
        # Use the LangChain wrapper's delete_index method
        # This safely handles "not found" errors
        vector_store.delete_index(index_name=index_name)
        print(
            f"Index '{index_name}' deletion acknowledged or index did not exist."
        )
    except Exception as e:
        print(f"Error during index deletion: {e}")


# --- CORRECTED EXECUTION BLOCK ---
# This block will run when you execute `python vector_store.py`
if __name__ == "__main__":
    
    # We must import data_prep here, inside the main block
    # (This will still fail if data_prep.py is not in the same folder)
    try:
        from data_prep import get_raw_weather_data, create_and_chunk_documents
    except ModuleNotFoundError:
        print("❌ ERROR: `data_prep.py` not found in the current directory.")
        print("Please ensure all .py files are in the same folder.")
        sys.exit(1)

    # 1. Prepare data
    raw_data = get_raw_weather_data()
    chunks = create_and_chunk_documents()

    # 2. Get the vector store connection
    print("\n--- Connecting to Vector Store ---")
    vector_store = get_opensearch_vector_store()
    
    # --- REINDEXING LOGIC ---
    print("\n--- Starting Full Reindexing Process ---")

    # 3. Delete the existing index
    delete_index(vector_store, OPENSEARCH_VECTOR_INDEX)

    # 4. Wait for 2 seconds (good practice for index deletion to settle)
    print("Waiting 2 seconds for index to delete...")
    time.sleep(2)

    # 5. Recreate and index documents
    # The OpenSearchVectorSearch object will automatically
    # create the index again when 'add_documents' is called.
    index_documents(vector_store, chunks)

    print("--- Full Reindexing Complete ---")