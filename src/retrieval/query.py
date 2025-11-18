import os
import sys
from typing import List, Optional, Dict

# --- Setup Paths ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from config.config import (
    OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, 
    VECTOR_FIELD_NAME, LLM_MODEL, K_VALUE
)

def get_opensearch_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
        request_timeout=30
    )

# --- API-Compatible Retriever Function ---
def simple_hybrid_retriever(
    query_text: str, 
    client: OpenSearch, 
    embeddings: GoogleGenerativeAIEmbeddings, 
    k: int = K_VALUE,
    filters: Optional[Dict] = None
):
    """
    Performs a hybrid search and returns a list of LangChain Documents.
    This function is designed to be called by the FastAPI endpoint.
    """
    try:
        vector_store = OpenSearchVectorSearch(
            index_name=INDEX_NAME,
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL,
            client=client,
            vector_field=VECTOR_FIELD_NAME,
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        
        # Note: The 'pre_filter' argument in LangChain's OpenSearch implementation 
        # maps to the OpenSearch/Elasticsearch 'filter' clause.
        results = vector_store.similarity_search(
            query=query_text, 
            k=k, 
            search_type="script_scoring", # Often more robust for hybrid
            pre_filter=filters
        )
        return results
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return []

# --- Standalone Generation Logic (for CLI) ---
async def get_llm_response(query: str, context: str):
    """Stub for LLM generation using the context."""
    # In production, call Gemini API here with the context
    return f"Based on the {len(context.splitlines())} chunks retrieved, here is an analysis for '{query}'..."

async def run_similarity_search(query_text, k=K_VALUE, pre_filter=None):
    """
    Wrapper that performs retrieval AND generation (for CLI usage).
    """
    client = get_opensearch_client()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    docs = simple_hybrid_retriever(query_text, client, embeddings, k, pre_filter)
    
    if not docs:
        return "No documents found."
        
    # Format context for LLM
    context_str = "\n".join([
        f"[Doc from {d.metadata.get('file_source_tag')}] {d.page_content}" 
        for d in docs
    ])
    
    return await get_llm_response(query_text, context_str)

# --- Main Execution (CLI Test) ---
if __name__ == "__main__":
    import asyncio
    async def main():
        print("--- Testing Retrieval ---")
        # Test 1: Filter by File Tag (IBM)
        print("\nQuery 1: IBM Risks")
        f1 = {"term": {"metadata.file_source_tag": "ibm"}}
        res1 = await run_similarity_search("What are the H1B risks?", pre_filter=f1)
        print(res1)
        
        # Test 2: Filter by Risk Category
        print("\nQuery 2: Contested Industry")
        f2 = {"term": {"metadata.risk_category": "Government Restrictions"}}
        res2 = await run_similarity_search("Details on contested industry?", pre_filter=f2)
        print(res2)

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Execution failed: {e}")