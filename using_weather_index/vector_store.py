import os
import sys
import time
from langchain_community.vectorstores import OpenSearchVectorSearch
from typing import List
from langchain_core.documents import Document
# Imports required for standalone execution block:
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- Configuration (Centralized or using Environment Variables) ---
OPENSEARCH_URL = "http://localhost:9200"
OPENSEARCH_VECTOR_INDEX = "weather-vector-store"
# NOTE: MiniLM-L6-v2 dimension is 384
EMBEDDING_DIMENSION = 384
VECTOR_FIELD = "vector_field"


def get_opensearch_vector_store(embeddings_client) -> OpenSearchVectorSearch:
    """
    Initializes the OpenSearch Vector Store connection using an
    already-initialized embeddings client.
    """

    print(
        f"Connecting to OpenSearch at {OPENSEARCH_URL} and index '{OPENSEARCH_VECTOR_INDEX}'..."
    )

    # Initialize the vector store connection
    vector_store = OpenSearchVectorSearch(
        index_name=OPENSEARCH_VECTOR_INDEX,
        # CRITICAL: Use the client passed into the function
        embedding_function=embeddings_client,
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

    # --- CRITICAL SANITY CHECK FOR 'DICT' OBJECT ERROR ---
    safe_documents = []
    for i, doc in enumerate(documents):
        # The logic to prevent the 'dict' object error
        if not isinstance(doc, Document) or not isinstance(doc.page_content, str) or not doc.page_content.strip():
            print(f"⚠️ Document skipped at index {i}. Invalid type or empty content.")
            continue

        safe_documents.append(doc)

    if not safe_documents:
        print("❌ ERROR: No safe documents to index. Check data_prep.py.")
        return []

    try:
        ids = vector_store.add_documents(safe_documents)
        print(
            f"Batch indexing complete. Total documents/chunks indexed: {len(ids)} (from {len(documents)} originals)"
        )
        return ids
    except Exception as e:
        print(f"❌ ERROR during indexing: {e}")
        raise e


def delete_index(vector_store: OpenSearchVectorSearch, index_name: str):
    """Deletes the specified OpenSearch index if it exists."""
    print(f"Attempting to delete index '{index_name}'...")
    try:
        # Use the LangChain wrapper's delete_index method
        vector_store.delete_index(index_name=index_name)
        print(
            f"Index '{index_name}' deletion acknowledged or index did not exist."
        )
    except Exception as e:
        print(f"Error during index deletion: {e}")


## 🚀 MAIN METHOD FOR STANDALONE EXECUTION
if __name__ == "__main__":

    # 1. Ensure required modules are available
    try:
        from data_prep import create_and_chunk_documents
        load_dotenv() # Load env vars for OpenSearch Auth (if needed)

    except ModuleNotFoundError:
        print("❌ ERROR: Required module 'data_prep.py' not found. Ensure all files are in the same directory.")
        sys.exit(1)

    # 2. Initialize Embeddings Client (HuggingFace)
    print("\n--- Initializing HuggingFace Embeddings for Standalone Run ---")
    standalone_embeddings_client = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 3. Prepare data and connect
    print("\n--- Preparing Data and Connecting to Vector Store ---")
    chunks = create_and_chunk_documents()
    vector_store = get_opensearch_vector_store(standalone_embeddings_client)

    # --- Full Reindexing Process ---
    print("\n--- Starting Full Reindexing Process ---")

    try:
        # 4. Delete the existing index
        delete_index(vector_store, OPENSEARCH_VECTOR_INDEX)

        # 5. Wait for 2 seconds
        print("Waiting 2 seconds for index to delete...")
        time.sleep(2)

        # 6. Recreate and index documents
        index_documents(vector_store, chunks)

        print("--- Full Reindexing Complete ---")
    except Exception as e:
        print(f"\n❌ CRITICAL INDEXING ERROR: {e}")
        print("Ensure OpenSearch is running and accessible.")