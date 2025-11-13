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
    Connects to OpenSearch and fetches all documents (up to 1000) from the 'weather' index.

    Returns:
        List[Dict[str, Any]]: A list of raw document source dictionaries, or an empty list on failure.
    """
    try:
        client = get_openseensearch_client()
        if not client.ping():
            print(f"❌ Connection failed for OpenSearch at {OPENSEARCH_URL}. Returning empty list.")
            return []

        # Define a match_all query to fetch all documents
        query = {
            "size": 1000,
            "query": {
                "match_all": {}
            }
        }

        response = client.search(index=INDEX_NAME, body=query)
        # Extract the '_source' (actual document data) from the hits
        documents = [hit['_source'] for hit in response['hits']['hits']]

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
    """
    Converts raw OpenSearch dictionaries into LangChain Document objects.
    Includes metadata sanitization to prevent downstream embedding/vector store errors.

    Args:
        raw_data: The list of raw dictionaries fetched from OpenSearch.

    Returns:
        List[Document]: A list of cleaned LangChain Document objects.
    """

    docs = []
    for i, item in enumerate(raw_data):
        # Use the "text" field as the main content
        content = item.get("text", "")

        # 1. Sanitize Metadata: Only keep simple key-value pairs
        # This is critical to prevent vector databases from failing on complex types (dicts/lists)
        safe_metadata = {}
        for k, v in item.items():
            if k == "text":
                continue # Skip the content field

            # Keep only simple types (strings, numbers, boolean)
            if isinstance(v, (str, int, float, bool)):
                safe_metadata[k] = v
            elif isinstance(v, (dict, list)):
                # CRITICAL: Skip complex types that can crash the embedder/vector store
                print(f"⚠️ Warning: Skipping complex metadata field '{k}' in document {i}.")
                continue
            else:
                 # Catch other odd types and convert them to string
                 safe_metadata[k] = str(v)


        if not isinstance(content, str) or not content.strip():
            print(f"⚠️ Warning: Document at index {i} has invalid/empty content. Skipping.")
            continue

        # 2. Create the Document with sanitized metadata
        docs.append(Document(page_content=content, metadata=safe_metadata))

    print(f"Transformed {len(docs)} OpenSearch items into LangChain Documents.")
    return docs

def create_and_chunk_documents(chunk_size: int = 250, chunk_overlap: int = 50) -> List[Document]:
    """
    Orchestrates the data loading from OpenSearch, conversion, and chunking process.

    Args:
        chunk_size: The maximum size of each text chunk.
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        List[Document]: A list of chunked and sanitized LangChain Document objects.
    """
    # 1. Load raw data from the external source (OpenSearch)
    raw_data = get_raw_weather_data()

    # 2. Convert to initial LangChain Documents and clean metadata
    initial_docs = _to_langchain_documents(raw_data)

    if not initial_docs:
        print("No documents retrieved from OpenSearch to chunk. Exiting data preparation.")
        return []

    print(f"Applying chunking (size={chunk_size}, overlap={chunk_overlap})...")

    # 3. Initialize the text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    # 4. Perform chunking
    chunked_docs = text_splitter.split_documents(initial_docs)

    # --- FINAL SANITIZATION STEP (Guaranteed robustness) ---
    final_safe_chunks = []
    for i, doc in enumerate(chunked_docs):
        # Ensure page_content is explicitly a string just before passing to embedding
        safe_content = str(doc.page_content)

        # Recreate the Document object to clean out any hidden complex internal types
        # that might have been introduced during chunking, or to confirm data types.
        safe_doc = Document(
            page_content=safe_content,
            metadata=doc.metadata # Metadata is assumed clean from _to_langchain_documents
        )
        final_safe_chunks.append(safe_doc)

    chunked_docs = final_safe_chunks
    # --- END FINAL SANITIZATION ---

    print(f"Final output: {len(chunked_docs)} document chunks ready.")
    return chunked_docs # Return the aggressively cleaned list

if __name__ == "__main__":
    # Example usage when running this file directly
    chunks = create_and_chunk_documents(chunk_size=100, chunk_overlap=20)
    if chunks:
        print("\n--- Example Chunk Retrieved from OpenSearch ---")
        print(f"Content: {chunks[0].page_content}")
        print(f"Metadata: {chunks[0].metadata}")