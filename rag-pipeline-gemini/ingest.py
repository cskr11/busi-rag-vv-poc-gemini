# ingest.py

import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions # Import OpenSearch client for deletion logic

# CORRECTED IMPORTS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, DATA_FILE

# --- Helper Functions ---

def get_opensearch_client():
    """Initializes and returns the raw OpenSearch client."""
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
        # Ensure client is ready to make requests
        request_timeout=30
    )

def delete_existing_index(client, index_name):
    """Deletes the OpenSearch index if it exists."""
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            print(f"-> Successfully deleted existing index: {index_name}")
        else:
            print(f"-> Index {index_name} does not exist. Skipping deletion.")
    except opensearch_exceptions.NotFoundError:
        print(f"-> Index {index_name} not found during deletion (expected).")
    except Exception as e:
        print(f"Error during index deletion: {e}")

def index_exists(client, index_name):
    """Checks if the OpenSearch index exists."""
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"Error checking index existence: {e}")
        return False

# --- Core Ingestion Logic ---

def create_sample_data(file_path):
    """Creates a dummy JSON file for demonstration purposes."""
    print(f"-> Creating sample data file: {file_path}")
    sample_data = [
        {"content": "The standard height for fall protection in construction is typically 6 feet (1.8 meters) above a lower level, but this can vary by industry and regulation."},
        {"content": "OpenSearch is an open-source search and analytics suite, forked from Elasticsearch and Kibana. It supports vector search through its k-NN plugin for high-performance retrieval."},
        {"content": "LangChain is a framework for developing applications powered by language models. It provides abstractions for components like vector stores, retrievers, and LLMs, simplifying RAG setup."},
        {"content": "Google's embedding models, like text-embedding-004, are highly effective for capturing the semantic meaning of text and powering similarity search in RAG pipelines."}
    ]
    with open(file_path, 'w') as f:
        json.dump(sample_data, f, indent=4)
    print("-> Sample data created successfully.")


def ingest_data_to_opensearch(docs_path, force_reindex=False):
    """
    Handles data ingestion, with an option to force a full reindex.
    """
    os_client = get_opensearch_client()

    if index_exists(os_client, INDEX_NAME):
        if force_reindex:
            print("--- FORCE REINDEX: Deleting existing index and reloading. ---")
            delete_existing_index(os_client, INDEX_NAME)
        else:
            print(f"--- Index {INDEX_NAME} already exists. Skipping ingestion. ---")
            return

    print(f"\n-> Starting data ingestion process into index: {INDEX_NAME}")

    # 1. Load Data and Prepare Documents
    try:
        with open(docs_path, 'r') as f:
            data = json.load(f)

        documents = [Document(page_content=item["content"], metadata={"source": docs_path, "doc_id": i}) for i, item in enumerate(data)]

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data from {docs_path}: {e}")
        return

    # 2. Split Documents (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    print(f"   - Split {len(documents)} source docs into {len(docs)} chunks.")

    # 3. Initialize Embeddings (Creating the Vector Embeddings)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"   - Initialized Google Embeddings model: {EMBEDDING_MODEL}")

    # 4. Create OpenSearch Vector Store and Index (Storing Embeddings)
    try:
        # from_documents handles embedding the chunks and uploading them to OpenSearch
        OpenSearchVectorSearch.from_documents(
            docs,
            embeddings,
            opensearch_url=OPENSEARCH_URL,
            index_name=INDEX_NAME,
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        print(f"-> Successfully indexed {len(docs)} chunks into OpenSearch index: {INDEX_NAME}")
    except Exception as e:
        print(f"Error connecting to or indexing in OpenSearch at {OPENSEARCH_URL}. Is OpenSearch running? Error: {e}")

if __name__ == "__main__":
    create_sample_data(DATA_FILE)

    # To run a full reindex (delete and load):
    # ingest_data_to_opensearch(DATA_FILE, force_reindex=True)

    # To load only if the index doesn't exist (default behavior):
    ingest_data_to_opensearch(DATA_FILE, force_reindex=False)