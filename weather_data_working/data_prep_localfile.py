from typing import List, Dict, Any
import json # NEW: Import the json module for file reading
import os   # NEW: Import os for file path management
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Local File Configuration ---
# DEFINE THE LOCAL FILE PATH
JSON_FILE_PATH = "data/knowledge-base.json" 

# --- 1. Data Source: Local File Connector ---

# REMOVE: get_opensearch_client is no longer needed

def get_raw_weather_data() -> List[Dict[str, Any]]:
    """
    Reads and loads data from the local JSON file into a Python list of dictionaries.

    Returns:
        List[Dict[str, Any]]: A list of raw document dictionaries, or an empty list on failure.
    """
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ Error: Local file not found at {JSON_FILE_PATH}. Returning empty list.")
        return []

    try:
        # Open and load the JSON file content
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            # json.load() deserializes the JSON file object directly into a Python list/dict
            documents = json.load(f)

        # Ensure the loaded data is in the expected list format
        if not isinstance(documents, list):
            print(f"⚠️ Warning: JSON file loaded as type {type(documents)}. Wrapping in a list.")
            documents = [documents] if documents else []

        print(f"Fetched {len(documents)} documents from local file '{JSON_FILE_PATH}'.")
        return documents

    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON file '{JSON_FILE_PATH}': {e}")
        return []
    except Exception as e:
        print(f"❌ An error occurred during local file reading: {e}")
        return []

# --- 2. Internal Transformation Function ---
# NOTE: This function remains the same as it correctly handles the 
# List[Dict[str, Any]] format returned by the new get_raw_weather_data.

def _to_langchain_documents(raw_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Converts raw Python dictionaries into LangChain Document objects.
    Includes metadata sanitization to prevent downstream embedding/vector store errors.

    Args:
        raw_data: The list of raw dictionaries fetched from the data source.

    Returns:
        List[Document]: A list of cleaned LangChain Document objects.
    """

    docs = []
    for i, item in enumerate(raw_data):
        # Use the "text" field as the main content (assuming the JSON structure uses "text")
        content = item.get("text", "")

        # 1. Sanitize Metadata: Only keep simple key-value pairs
        safe_metadata = {}
        for k, v in item.items():
            if k == "text":
                continue # Skip the content field

            # Keep only simple types (strings, numbers, boolean)
            if isinstance(v, (str, int, float, bool)):
                safe_metadata[k] = v
            elif isinstance(v, (dict, list)):
                # Skip complex types
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

    print(f"Transformed {len(docs)} data items into LangChain Documents.")
    return docs

def create_and_chunk_documents(chunk_size: int = 250, chunk_overlap: int = 50) -> List[Document]:
    """
    Orchestrates the data loading from the local file, conversion, and chunking process.

    Args:
        chunk_size: The maximum size of each text chunk.
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        List[Document]: A list of chunked and sanitized LangChain Document objects.
    """
    # 1. Load raw data from the external source (Local JSON file)
    raw_data = get_raw_weather_data()

    # 2. Convert to initial LangChain Documents and clean metadata
    initial_docs = _to_langchain_documents(raw_data)

    if not initial_docs:
        print("No documents retrieved from the data source to chunk. Exiting data preparation.")
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
        # Ensure page_content is explicitly a string
        safe_content = str(doc.page_content)

        # Recreate the Document object to confirm data types.
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
        print("\n--- Example Chunk Retrieved from Local File ---")
        print(f"Content: {chunks[0].page_content}")
        print(f"Metadata: {chunks[0].metadata}")