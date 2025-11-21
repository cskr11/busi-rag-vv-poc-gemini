import os
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any
import httpx

# --- Setup Paths ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from config.config import (
    OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL,
    VECTOR_FIELD_NAME, LLM_MODEL, K_VALUE
)

# --- LLM Client Setup ---
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
    """Handles API calls with exponential backoff."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
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
                await asyncio.sleep(2 ** attempt)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    raise Exception("API call failed after all retry attempts.")

# --- Context-Aware Memory Function ---

async def condense_query(history: List[Dict[str, str]], new_prompt: str) -> str:
    """Condenses conversation history into a standalone search query."""
    if not history:
        return new_prompt

    history_str = "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history])
    condensation_prompt = (
        "You are a helpful AI assistant. Your task is to condense the provided conversation history "
        "and the final user question into a single, precise, standalone search query. "
        "Example: If history mentions 'IBM' and the new question is 'What about their sanctions risks?', "
        "output: 'IBM sanctions risks'. Generate only the search query string."
    )
    user_content = f"--- CONVERSATION HISTORY ---\n{history_str}\n---\nUser's new question: {new_prompt}"
    payload = {
        "contents": [{"parts": [{"text": user_content}]}],
        "systemInstruction": {"parts": [{"text": condensation_prompt}]},
    }
    try:
        response_json = await _api_call_with_backoff(payload, API_BASE_URL)
        return response_json['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Query Condensation failed ({e}). Falling back to original prompt.")
        return new_prompt

# --- Retrieval Functions (Enhanced for Strictness) ---

def simple_hybrid_retriever(
    search_query: str,
    client: OpenSearch,
    embeddings: GoogleGenerativeAIEmbeddings,
    k: int = K_VALUE,
    filters: Optional[Dict] = None
) -> List[Document]:
    """
    Performs Hybrid Search with Strict Phrase Boosting.
    Prioritizes documents that contain the exact search phrase.
    """
    print(f" simple_hybrid_retriever called with query: {search_query}, for filters: {filters}, k-results: {k}")
    # 1. Vector Search (Concepts)
    vector_results = []
    try:
        vector_store = OpenSearchVectorSearch(
            index_name=INDEX_NAME,
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL,
            client=client,
            vector_field=VECTOR_FIELD_NAME,
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        vector_results = vector_store.similarity_search(
            query=search_query,
            k=k,
            search_type="script_scoring",
            pre_filter=filters
        )
    
    except Exception as e:
        print(f"Vector search failed: {e}")

    # 2. Keyword Search (Strict Matches)
    keyword_results = []
    try:
        dsl_query = {
            "query": {
                "bool": {
                    "should": [
                        # Broad Match (matches individual words anywhere)
                        {
                            "multi_match": {
                                "query": search_query,
                                "fields": [
                                    "page_content", 
                                    "metadata.company_name",
                                    "metadata.aliases",
                                    "metadata.source_id^5" # High priority for IDs
                                ],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        # Strict Phrase Match (matches the exact sequence of words)
                        # This fixes the "not strict enough" issue by heavily boosting exact phrases
                        {
                            "match_phrase": {
                                "page_content": {
                                    "query": search_query,
                                    "boost": 10.0 
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": k
        }

        if filters:
            # Basic filter implementation
            for key, value in filters.get("term", {}).items():
                dsl_query["query"]["bool"]["filter"] = [{"term": {key: value}}]

        raw_response = client.search(index=INDEX_NAME, body=dsl_query)
        
        for hit in raw_response['hits']['hits']:
            source = hit['_source']
            doc = Document(
                page_content=source.get('page_content', ''),
                metadata=source.get('metadata', {})
            )
            keyword_results.append(doc)
            
    except Exception as e:
        print(f"Keyword search failed: {e}")

    # 3. Merge & Deduplicate (Preferring Keyword/Phrase Matches)
    seen_ids = set()
    final_results = []

    def get_doc_sig(d):
        # Create unique signature based on source ID + snippet
        return f"{d.metadata.get('source_id', 'N/A')}-{d.page_content[:50]}"

    # Add strict keyword results first
    for doc in keyword_results:
        sig = get_doc_sig(doc)
        if sig not in seen_ids:
            seen_ids.add(sig)
            final_results.append(doc)

    # Fill remainder with vector results
    for doc in vector_results:
        if len(final_results) >= k * 2: # Cap total results
            break
        sig = get_doc_sig(doc)
        if sig not in seen_ids:
            seen_ids.add(sig)
            final_results.append(doc)

    return final_results

# --- Generation Logic ---

async def get_llm_response(full_prompt: str, context: str, doc_count: int): # <-- Accept doc_count
    if not API_KEY:
        return "LLM API Key missing."

    # --- RESTRUCTURED SYSTEM INSTRUCTION ---
    system_instruction = (
        "You are a risk analysis assistant. Your primary task is to consolidate ALL provided context documents into a comprehensive report. "
        "The user wants ALL risks found (Total documents retrieved: "
        f"{doc_count}). Answer based ONLY on the provided context. "
        "Your final response MUST be structured into exactly three sections: "
        
        "### 1. Risk Summary in a Nutshell"
        "Generate a brief, high-level summary paragraph. State the number of distinct risk findings found and list all major Risk Categories present."
        
        "### 2. Matched Documents (Structured List)"
        "For EACH document provided in the '--- METADATA LIST START ---' block, create a clear, itemized markdown list containing only the following fields in this exact order: Entity Name, Risk Category, Risk Subcategory, Priority Value, and Source ID. Extract these values directly from the Metadata blocks."
        
        "### 3. Retrieved Content"
        "List the full, raw content of EACH risk document found in the '--- FULL PAGE CONTENT LIST START ---' block. For each document, prepend the content with a bolded header like: **[Document X Full Content]**"
        
        "\n\nBegin your response immediately, starting with '### 1. Risk Summary in a Nutshell'."
    )
    # --- END RESTRUCTURED SYSTEM INSTRUCTION ---

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
        return f"LLM generation failed: {e}"

async def run_context_aware_rag(new_prompt: str, history: Optional[List] = None, k: int = K_VALUE):
    """CLI Entry point"""
    client = get_opensearch_client()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    search_query = await condense_query(history or [], new_prompt)
    docs = simple_hybrid_retriever(search_query, client, embeddings, k)
    
    context_str = "\n".join([
        f"[Doc SourceID: {d.metadata.get('source_id', 'N/A')}] {d.page_content.strip().replace(chr(10), ' ')}"
        for d in docs
    ])
    return await get_llm_response(new_prompt, context_str, k)

if __name__ == "__main__":
    # CLI Test
    asyncio.run(run_context_aware_rag("What are the Adversarial Supply Chain risks?"))