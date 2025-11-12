# data_prep.py
from typing import List, Dict, Any
from opensearchpy import OpenSearch, exceptions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- OpenSearch Connection Configuration ---
HOST = 'localhost'
PORT = 9200
OPENSEARCH_URL = f"http://{HOST}:{PORT}"
INDEX_NAME = "weather"
# Assuming default credentials for local POC container
AUTH = ('admin', 'admin')

# --- 1. Data Source: OpenSearch Connector ---

def get_openseensearch_client() -> OpenSearch:
    """Initializes and returns the OpenSearch client connection."""
    client = OpenSearch(
        hosts=[{'host': HOST, 'port': PORT}],
        http_auth=AUTH,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client

def get_raw_weather_data() -> List[Dict[str, Any]]:
    """
    Connects to OpenSearch and fetches all documents from the 'weather' index.
    This replaces the hardcoded list.
    """
    try:
        client = get_openseensearch_client()
        if not client.ping():
            print(f"❌ Connection failed for OpenSearch at {OPENSEARCH_URL}. Returning empty list.")
            return []

        # Use a match_all query to retrieve all documents
        query = {
            "size": 1000,  # Max number of documents to retrieve in one call
            "query": {
                "match_all": {}
            }
        }

        response = client.search(index=INDEX_NAME, body=query)

        documents = []
        for hit in response['hits']['hits']:
            # The actual document data is under the '_source' key
            documents.append(hit['_source'])

        print(f"Fetched {len(documents)} documents from OpenSearch index '{INDEX_NAME}'.")
        return documents

    except exceptions.NotFoundError:
        print(f"❌ Index '{INDEX_NAME}' not found in OpenSearch. Please index the data first.")
        return []
    except Exception as e:
        print(f"❌ An error occurred during OpenSearch data fetching: {e}")
        return []

# --- 2. Internal Transformation Function ---

def _to_langchain_documents(raw_data: List[Dict[str, Any]]) -> List[Document]:
    """Converts a list of raw data dicts into a list of LangChain Document objects."""

    docs = []
    for item in raw_data:
        # Extract the content for embedding (the 'text' field is assumed to be present)
        content = item.get("text", "")
        # All other fields become metadata
        metadata = {k: v for k, v in item.items() if k != "text"}

        if content:
            docs.append(Document(page_content=content, metadata=metadata))

    print(f"Transformed {len(docs)} OpenSearch items into LangChain Documents.")
    return docs

# --- 3. Public Processing Function (Chunking) ---

def create_and_chunk_documents(chunk_size: int = 250, chunk_overlap: int = 50) -> List[Document]:
    """
    Orchestrates the data loading from OpenSearch, conversion, and chunking process.
    Returns the final list of chunks ready for indexing/retrieval.
    """
    # 3.1 Load and Convert
    raw_data = get_raw_weather_data()
    initial_docs = _to_langchain_documents(raw_data)

    if not initial_docs:
        print("No documents retrieved from OpenSearch to chunk. Exiting data preparation.")
        return []

    # 3.2 Define the Chunking Strategy
    print(f"Applying chunking (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # 3.3 Apply chunking
    chunked_docs = text_splitter.split_documents(initial_docs)

    print(f"Final output: {len(chunked_docs)} document chunks ready.")
    return chunked_docs

# Example use
if __name__ == "__main__":
    chunks = create_and_chunk_documents(chunk_size=100, chunk_overlap=20)
    if chunks:
        print("\n--- Example Chunk Retrieved from OpenSearch ---")
        print(f"Content: {chunks[0].page_content}")
        print(f"Metadata: {chunks[0].metadata}")
