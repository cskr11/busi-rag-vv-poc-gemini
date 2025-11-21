import os
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
import httpx
from operator import itemgetter

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
        # Use a secondary filter to remove conversational text if condensation fails
        return response_json['candidates'][0]['content']['parts'][0]['text'].strip().replace('"', '')
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
    Performs Hybrid Search with Enhanced Boosting and Filtering.
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

    # 2. Keyword Search (Strict Matches with Enhanced Boosting)
    keyword_results = []
    try:
        # NOTE: Boosted 'metadata.priority_level' to help keyword search align with query intent (e.g., "high priority")
        dsl_query = {
            "query": {
                "bool": {
                    "should": [
                        # Broad Match with standard boosting
                        {
                            "multi_match": {
                                "query": search_query,
                                "fields": [
                                    "page_content",
                                    "metadata.entity_name",
                                    "metadata.aliases",
                                    "metadata.priority_level^2", # Enhanced: Boost priority keywords
                                    "metadata.source_id^5"
                                ],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        # Strict Phrase Match (Heavy boosting for exact sequence)
                        {
                            "match_phrase": {
                                "page_content": {
                                    "query": search_query,
                                    "boost": 15.0 # Enhanced: Increased boost for strict phrase
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
            # Simple filter implementation: only 'term' supported for brevity
            if filters.get("term"):
                # Using 'filter' clause in 'bool' query for accurate filtering
                dsl_query["query"]["bool"]["filter"] = [
                    {"term": {k: v}} for k, v in filters["term"].items()
                ]

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
    # Only add up to K_VALUE documents total (keyword + vector) for a cleaner context window
    for doc in vector_results:
        if len(final_results) >= k:
            break
        sig = get_doc_sig(doc)
        if sig not in seen_ids:
            seen_ids.add(sig)
            final_results.append(doc)

    return final_results

# --- Generation Logic ---

async def get_llm_response(full_prompt: str, docs: List[Document]):
    """Generates the LLM response using strictly formatted context."""
    if not API_KEY:
        return "LLM API Key missing."

    # --- 1. Post-Retrieval Sorting/Refinement ---
    # Sort documents by priority_number descending (High risk first)
    # Default to 0 for missing priority_number
    sorted_docs = sorted(
        docs,
        key=lambda d: d.metadata.get('priority_number', 0),
        reverse=True
    )

    doc_count = len(sorted_docs)

    # --- 2. Create Structured Context Blocks (FIXED FOR CLEAN CONTENT) ---
    metadata_block = []
    content_block = []

    for i, d in enumerate(sorted_docs):
        # Enforce metadata structure for easy LLM extraction
        metadata_block.append(
            f"--- DOCUMENT {i+1} METADATA ---\n"
            f"Entity Name: {d.metadata.get('entity_name', 'N/A')}\n"
            f"Risk Category: {d.metadata.get('risk_category', 'N/A')}\n"
            f"Risk Subcategory: {d.metadata.get('risk_subcategory', 'N/A')}\n"
            f"Priority Value: {d.metadata.get('priority_level', 'N/A')} ({d.metadata.get('priority_number', 'N/A')})\n"
            f"Source ID: {d.metadata.get('source_id', 'N/A')}\n"
        )

        # Prepare content block with clear header and document text
        # FIX: Ensure content is appended cleanly with just a newline separator,
        # and strip unnecessary surrounding whitespace from the page content.
        content_block.append(
            f"**[Document {i+1} Full Content]**\n"
            f"{d.page_content.strip()}\n" # Removed extra \n escaping and ensured strip()
        )

    # Join the context parts
    # FIX: Use simple '\n' joiner and clean up surrounding f-string newlines
    context_str = (
        f"--- METADATA LIST START (TOTAL: {doc_count}) ---\n"
        f"{'\n'.join(metadata_block)}\n"
        f"--- METADATA LIST END ---\n\n"
        f"--- FULL PAGE CONTENT LIST START (TOTAL: {doc_count}) ---\n"
        f"{'\n'.join(content_block)}"
        f"\n--- FULL PAGE CONTENT LIST END ---\n"
    )

    # --- 3. Restructured System Instruction ---
    system_instruction = (
        "You are a risk analysis assistant. Your primary task is to consolidate ALL provided context documents into a comprehensive risk report. "
        "The retrieved documents are pre-sorted by risk priority (highest risk first). "
        "Answer based ONLY on the provided context. "
        "Your final response MUST be structured into exactly three sections: "

        "### 1. Risk Summary in a Nutshell"
        "Generate a brief, high-level summary paragraph. State the **total number of distinct risk findings** found and list **ALL major Risk Categories** present in the documents, making specific mention of the highest priority risks." # 🎯 EDIT 1: Add a directive on sorting/priority
        
        "### 2. Matched Documents (Structured List)"
        "For EACH document provided in the '--- METADATA LIST START ---' block, create a clear, itemized markdown list. The items in the list MUST contain the following fields in this exact order: **Entity Name**, **Risk Category**, **Risk Subcategory**, **Priority Level**, **Priority Number**, and **Source ID**. "
        "Extract the Level and Number from the Metadata blocks and ensure the output is presented in descending order of Priority Number." # <-- MODIFIED INSTRUCTION

        "### 3. Retrieved Content"
        "List the full, raw content of EACH risk document found in the '--- FULL PAGE CONTENT LIST START ---' block. Maintain the original formatting and bolded headers (e.g., **[Document 1 Full Content]**). Do not alter this content in any way." # 🎯 EDIT 3: Stronger constraint on content integrity

        "\n\nBegin your response immediately, starting with '### 1. Risk Summary in a Nutshell'."
    )
    # --- END RESTRUCTURED SYSTEM INSTRUCTION ---

    prompt_with_context = (
        f"--- CONTEXT START ---\n{context_str}\n--- CONTEXT END ---\n\n"
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

    # Retrieve documents
    docs = simple_hybrid_retriever(search_query, client, embeddings, k)

    # Pass the full document list to the generation logic
    return await get_llm_response(new_prompt, docs)

if __name__ == "__main__":
    # CLI Test
    asyncio.run(run_context_aware_rag("What are the Adversarial Supply Chain risks?"))