import os
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
# FIX: Import httpx at the top level to ensure it's always bound
import httpx

# --- Setup Paths ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from config.config import (
    OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL,
    VECTOR_FIELD_NAME, LLM_MODEL, K_VALUE
)

# --- LLM Client Setup ---
# The API key must be provided via the GOOGLE_API_KEY environment variable.
API_KEY = os.getenv("GOOGLE_API_KEY", "")
API_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={API_KEY}"

def get_opensearch_client():
    """Initializes and returns the raw OpenSearch client."""
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
        request_timeout=30
    )

async def _api_call_with_backoff(payload: Dict, url: str) -> Dict:
    """
    Handles API calls with exponential backoff for reliability.
    Returns Dict[str, Any] or raises an exception.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # httpx is now imported at the top level
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    headers={'Content-Type': 'application/json'},
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API Error {e.response.status_code}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                # If last attempt fails, raise the error
                raise
        except Exception:
            # Catch other exceptions (e.g., connection errors) and re-raise if last attempt
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Connection Error. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

    # FIX: Explicitly raise an exception if loop finishes without returning (should be unreachable
    # due to the 'raise' inside the loop, but satisfies type checker)
    raise Exception("API call failed after all retry attempts.")


# --- Context-Aware Memory Function ---

async def condense_query(history: List[Dict[str, str]], new_prompt: str) -> str:
    """
    Uses the LLM to condense the chat history and new prompt into a single,
    context-aware search query.
    """
    if not history:
        # If no history, the new prompt is the search query
        return new_prompt

    history_str = "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history])

    condensation_prompt = (
        "You are a helpful AI assistant. Your task is to condense the provided conversation history "
        "and the final user question into a single, precise, standalone search query. "
        "The search query must be specific and include all relevant context from the history. "
        "Example: If history mentions 'IBM' and the new question is 'What about their sanctions risks?', "
        "the output must be: 'IBM sanctions risks'. "
        "Generate only the search query string, nothing else."
    )

    user_content = f"--- CONVERSATION HISTORY ---\n{history_str}\n---\nUser's new question: {new_prompt}"

    payload = {
        "contents": [{"parts": [{"text": user_content}]}],
        "systemInstruction": {"parts": [{"text": condensation_prompt}]},
    }

    try:
        response_json = await _api_call_with_backoff(payload, API_BASE_URL)

        condensed_query = response_json['candidates'][0]['content']['parts'][0]['text']
        print(f"✅ Query Condensed: {condensed_query}")
        return condensed_query.strip()
    except Exception as e:
        print(f"❌ Query Condensation failed ({e}). Falling back to original prompt.")
        return new_prompt


# --- Retrieval Functions ---
def simple_hybrid_retriever(
    search_query: str,
    client: OpenSearch,
    embeddings: GoogleGenerativeAIEmbeddings,
    k: int = K_VALUE,
    filters: Optional[Dict] = None
) -> List[Document]:
    """
    Performs a true Hybrid Search (Vector + Keyword) to capture both 
    semantic meaning and exact matches (like IDs or specific Entity Names).
    """
    # 1. Setup Vector Store for Semantic Search
    vector_store = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
        client=client,
        vector_field=VECTOR_FIELD_NAME,
        client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
    )

    # 2. Run Vector Search (Captures "concepts")
    try:
        vector_results = vector_store.similarity_search(
            query=search_query,
            k=k,
            search_type="script_scoring",
            pre_filter=filters
        )
    except Exception as e:
        print(f"Vector search failed: {e}")
        vector_results = []

    # 3. Run Keyword Search (Captures "exact matches" like Source IDs or Names)
    # We search across page_content AND metadata fields
    keyword_results = []
    try:
        dsl_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": search_query,
                                "fields": [
                                    "page_content^2",       # Boost content matches
                                    "metadata.source_id^3", # Boost ID matches heavily
                                    "metadata.company_name",
                                    "metadata.source_name"
                                ],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            },
            "size": k
        }

        # Apply Filters to Keyword Search if present
        if filters:
            # Convert simple term match to OpenSearch DSL filter
            # This is a basic implementation; complex filters might need recursion
            for key, value in filters.get("term", {}).items():
                dsl_query["query"]["bool"]["filter"] = [{"term": {key: value}}]

        raw_response = client.search(index=INDEX_NAME, body=dsl_query)
        
        # Convert Raw OpenSearch Hits to LangChain Documents
        for hit in raw_response['hits']['hits']:
            source = hit['_source']
            doc = Document(
                page_content=source.get('page_content', ''),
                metadata=source.get('metadata', {})
            )
            keyword_results.append(doc)
            
    except Exception as e:
        print(f"Keyword search failed: {e}")

    # 4. Merge & Deduplicate Results
    # We prioritize Keyword matches for ID lookups, then fill with Vector matches
    seen_ids = set()
    final_results = []

    # Helper to generate a unique signature for deduplication
    def get_doc_sig(d):
        return f"{d.metadata.get('source_id', '')}-{d.page_content[:50]}"

    # Add keyword results first (high precision)
    for doc in keyword_results:
        sig = get_doc_sig(doc)
        if sig not in seen_ids:
            seen_ids.add(sig)
            final_results.append(doc)

    # Add vector results next (high recall)
    for doc in vector_results:
        sig = get_doc_sig(doc)
        if sig not in seen_ids:
            seen_ids.add(sig)
            final_results.append(doc)

    return final_results[:k*2] # Return a slightly larger pool since we merged

# --- Standalone Generation Logic (for CLI) ---

async def get_llm_response(full_prompt: str, context: str):
    """
    Calls the LLM API to generate a grounded response based on the context.
    """
    if not API_KEY:
        return f"LLM API Key missing. Context retrieved ({len(context.splitlines())} chunks). Cannot generate response."

    # 1. System Prompt
    system_instruction = (
        "You are a sophisticated risk analysis assistant. Your goal is to answer the final user question "
        "concisely and accurately based ONLY on the provided context below. "
        "If the context is insufficient, state that you cannot answer based on the available data."
    )

    # 2. Prepare the combined prompt
    prompt_with_context = (
        f"--- CONTEXT START ---\n{context}\n--- CONTEXT END ---\n\n"
        f"FINAL USER QUESTION: {full_prompt}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_with_context}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }

    try:
        response_json = await _api_call_with_backoff(payload, API_BASE_URL)
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"LLM generation failed due to API error: {e}"


async def run_context_aware_rag(
    new_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    k: int = K_VALUE,
    pre_filter: Optional[Dict] = None
):
    """
    Orchestrates the context-aware RAG pipeline for CLI testing.
    """
    print(f"\n{'='*60}\nUSER PROMPT: '{new_prompt}'")
    print(f"HISTORY LENGTH: {len(history) if history else 0}")

    client = get_opensearch_client()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 1. Condense Query using history
    search_query = await condense_query(history or [], new_prompt)

    # 2. Retrieve Documents using the condensed query
    docs = simple_hybrid_retriever(search_query, client, embeddings, k, pre_filter)

    if not docs:
        return "No relevant risk documents found after applying filters."

    # 3. Format Context
    context_str = "\n".join([
        f"[Doc {i+1} from {d.metadata.get('file_source_tag', 'N/A')}] {d.page_content.strip().replace('\n', ' ')}"
        for i, d in enumerate(docs)
    ])

    print(f"Retrieved {len(docs)} documents using search query: '{search_query}'")

    # 4. Generate Final Response
    return await get_llm_response(new_prompt, context_str)


# --- Main Execution (CLI Test) ---
if __name__ == "__main__":

    async def main():
        # Test Conversation History
        history1 = [
            {"role": "user", "text": "I need to analyze the risks associated with IBM."},
            {"role": "model", "text": "IBM has 96 total findings. What specific risk category should we focus on?"}
        ]

        # New prompt relies on "IBM" from history
        new_prompt1 = "What is the priority score for its Government Restrictions findings?"

        # We still apply the filter, but the search query is now much smarter
        filter1 = {"term": {"metadata.file_source_tag": "ibm"}}

        print("\n*** RUNNING CONTEXT-AWARE TEST 1 ***")
        response1 = await run_context_aware_rag(new_prompt1, history1, pre_filter=filter1)
        print("\nFINAL CONTEXT-AWARE RESPONSE:")
        print(response1)

    # Note: Requires GOOGLE_API_KEY set in environment or .env
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"CLI execution failed. Check API key and OpenSearch connection. Error: {e}")