import os
import sys
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from typing import Any, Dict

# Load environment variables (CRITICAL: Reads GOOGLE_API_KEY from .env)
load_dotenv()

# --- CONFIGURATION (COPIED from vector_store.py) ---
OPENSEARCH_URL = "http://localhost:9200"
OPENSEARCH_VECTOR_INDEX = "weather-vector-store"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
# -------------------------------------------------------------------


def get_opensearch_vector_store_standalone():
    """
    Initializes the embedding model and OpenSearch Vector Store connection.
    This function contains the authentication fix, now using SecretStr.
    """
    # 1. Retrieve the API Key
    api_key_str = os.getenv("GOOGLE_API_KEY")

    # 2. Key Check and Conversion (Handling the str -> SecretStr requirement)
    if not api_key_str:
        # Exit gracefully if key is missing
        print(
            "❌ ERROR: GOOGLE_API_KEY environment variable not found. Check your .env file."
        )
        sys.exit(1)

    # Convert the string into Pydantic's SecretStr
    api_key = SecretStr(api_key_str)

    print("\n[Standalone] Initializing Google Generative AI Embeddings...")

    # CRITICAL FIX: Explicitly pass the SecretStr object to the constructor
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, google_api_key=api_key
    )

    print(f"[Standalone] Connecting to OpenSearch at {OPENSEARCH_URL}...")

    # Initialize the vector store connection
    vector_store = OpenSearchVectorSearch(
        index_name=OPENSEARCH_VECTOR_INDEX,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        # Default configuration fields from vector_store.py
        vector_field="vector_field",
        text_field="text",
        engine="nmslib",
        space_type="l2",
        dim=768,
        m=16,
        ef_construction=200,
    )
    return vector_store


def delete_index_standalone(
    vector_store: OpenSearchVectorSearch, index_name: str
):
    """
    Deletes the specified OpenSearch index using the LANGCHAIN WRAPPER method.
    """
    print(f"[Standalone] Attempting to delete index '{index_name}'...")
    try:
        
        # --- THIS IS THE CORRECTED LINE ---
        # We call the delete_index method on the vector_store object itself,
        # not the raw client. This wrapper method handles ignoring 404 errors.
        vector_store.delete_index(index_name=index_name)
        # --- END OF CORRECTION ---

        print(
            f"[Standalone] Index '{index_name}' deletion acknowledged or index did not exist."
        )

    except Exception as e:
        print(f"[Standalone] Error during index deletion: {e}")


def main():
    """
    Connects to OpenSearch and executes the deletion task.
    """
    try:
        # Step 1: Get the connection object (Authenticates the embedding model here)
        vector_store = get_opensearch_vector_store_standalone()

        # Step 2: Execute the deletion
        delete_index_standalone(vector_store, OPENSEARCH_VECTOR_INDEX)

        print(
            f"\nCleanup complete. Index '{OPENSEARCH_VECTOR_INDEX}' is ready to be recreated."
        )

    except Exception as e:
        print(f"❌ Critical Error during standalone deletion.")
        print(f"Error details: {e}")


if __name__ == "__main__":
    main()