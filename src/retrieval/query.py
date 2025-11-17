import sys
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure the parent directory (src) is in the path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations (assuming they are set correctly)
from config.config import LLM_MODEL, OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, VECTOR_FIELD_NAME, K_VALUE

# Load environment variables if needed
load_dotenv()

# --- Helper Functions (From ingest.py) ---
def get_opensearch_client():
    """Initializes and returns the raw OpenSearch client."""
    # Assuming OPENSEARCH_URL is defined in config.py
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
        request_timeout=30
    )

# --- NEW: Entity Identification Helper (Updated for fuzzy matching) ---
def get_entity_filter(query: str) -> Optional[Dict[str, Any]]:
    """
    Checks the query for a known company name and returns a mandatory query_string filter
    for robust entity filtering using component words.
    """
    lower_query = query.lower()

    # Identify the target company name
    if "international business machines" in lower_query or "ibm" in lower_query:
        company_name = "International Business Machines Corporation"
    elif "myntra" in lower_query or "jabong" in lower_query:
        company_name = "MYNTRA JABONG INDIA PRIVATE LIMITED"
    else:
        return None

    # CRITICAL FIX: Use query_string *without* quotes to allow flexible matching of component words.
    # This solves the "fuzzy match" request.
    return {
        "query_string": {
            "query": company_name,
            "fields": ["metadata.company_name"],
            "default_operator": "AND" # All words must be present, but order doesn't matter
        }
    }

def simple_hybrid_retriever(query: str, client: OpenSearch, embeddings: GoogleGenerativeAIEmbeddings, k: int = K_VALUE) -> List[Document]:

    print("  - Starting simple hybrid search (KNN + Full Text Match)...")

    try:
        # 1. Generate Query Vector (Semantic Search Input)
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        print(f"Error generating embedding vector: {e}")
        return []

    # 2. Structured Filtering: Determine mandatory filters
    entity_filter = get_entity_filter(query)

    if entity_filter:
        mandatory_filters = [entity_filter]
        print(f"  - Applying mandatory fuzzy entity filter.")
    else:
        # If no specific entity is mentioned, apply the default filter to include all doc types
        mandatory_filters = [
            {"bool": {"should": [
                {"term": {"metadata.doc_type": "RiskFinding"}},
                {"term": {"metadata.doc_type": "ProfileSummary"}}
            ], "minimum_should_match": 1}}
        ]
        print("  - Applying default document type filter.")


    # 3. Build the Hybrid Query Body
    query_body = {
        "size": k,
        "query": {
            "bool": {
                # This filter is MANDATORY and restricts the search space first.
                "filter": mandatory_filters,

                # These are the scoring clauses (KNN + Text).
                "should": [
                    {
                        "knn": {
                            VECTOR_FIELD_NAME: {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    # Add a strong Lucene search to catch keyword matches
                    {
                        "query_string": {
                            "query": query,
                            "fields": ["page_content^2", "text^2", "metadata.company_name^3", "metadata.risk_subcategory^2"],
                            "default_operator": "OR"
                        }
                    }
                ],
                "minimum_should_match": 1 # Ensures at least one scoring clause matches
            }
        },
        "_source": ["page_content", "text", "metadata"]
    }

    # 4. Execute the Search and Format Results
    try:
        response = client.search(index=INDEX_NAME, body=query_body)

        docs = []
        for hit in response['hits']['hits']:
            source = hit['_source']

            # Safely retrieve content
            content = source.get('page_content')
            if content is None:
                content = source.get('text')

            if content is None:
                 continue

            metadata = source.get('metadata', {})
            docs.append(Document(page_content=content, metadata=metadata))

        print(f"  - Retrieved {len(docs)} documents using Entity-Filtered Hybrid Search.")
        return docs

    except opensearch_exceptions.NotFoundError:
        print(f"  - Index {INDEX_NAME} not found during hybrid search. Did you run ingest.py?")
        return []
    except Exception as e:
        print(f"  - Error during simple OpenSearch query: {e}")
        return []

# --- RAG SIMULATION LOGIC (Unchanged) ---

def run_rag_query(query: str, client: OpenSearch, embeddings: GoogleGenerativeAIEmbeddings):

    # 1. Retrieval: Using the simplified retriever
    retrieved_docs = simple_hybrid_retriever(query, client, embeddings)

    if not retrieved_docs:
        print("\n[AI ASSISTANT RESPONSE]\nNo relevant context was retrieved from OpenSearch.")
        return

    # 2. Context Stuffing
    context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # 3. Generation
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)

    SYSTEM_TEMPLATE = """
    You are a highly analytical risk assessment AI. Your role is to answer the user's question
    **based ONLY on the provided CONTEXT**.

    - Summarize the business name, the number of total findings (if available), and the highest priority risk found.
    - If a company has 'no detailed risk findings', explicitly state this.
    - Be professional, concise, and factual.

    CONTEXT:
    ---
    {context}
    ---
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    # Invoke the chain
    print(f"\n--- Running Query: '{query}' ---")
    response = chain.invoke({"context": context_text, "question": query})

    # 4. Output Results
    print("\n[AI ASSISTANT RESPONSE]")
    print(response.content)
    print("\n-------------------------------------")
    print("RELEVANT CONTEXT CHUNKS RETRIEVED:")
    for i, doc in enumerate(retrieved_docs):
        company_name = doc.metadata.get('company_name', 'N/A')
        doc_type = doc.metadata.get('doc_type', 'N/A')
        risk_sub = doc.metadata.get('risk_subcategory', 'N/A')
        print(f"[{i+1}] **{company_name}** ({doc_type}) | Subcategory: {risk_sub}")
        print(f"      Snippet: {doc.page_content[:150]}...")


# --- MAIN EXECUTION (Unchanged) ---

if __name__ == "__main__":

    os_client = get_opensearch_client()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, request_options={"timeout": 60})

    print("="*50)
    print(f"STARTING LIVE QUERY SIMULATION (K={K_VALUE}, Index='{INDEX_NAME}')")
    print("="*50)

    # Simple queries that rely on KNN and fuzzy text matching:
    simple_test_queries = [
        "Tell me about the risks for IBM.",
        "What risk does Myntra have regarding Hong Kong business?",
        "Are there any risks related to visas for International Business Machines?",
        "What is the overall profile score for MYNTRA JABONG INDIA PRIVATE LIMITED?",
        "Is there a high priority risk for International Business Machines?"
    ]

    for q in simple_test_queries:
        run_rag_query(q, os_client, embeddings)
        print("\n" + "-"*50 + "\n")