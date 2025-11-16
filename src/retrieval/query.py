import sys
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
from typing import List, Dict, Any

# Ensure the parent directory (src) is in the path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration and essential functions (assuming config.py is in src/config)
from config.config import INDEX_NAME, EMBEDDING_MODEL, LLM_MODEL 
from ingestion.ingest import get_opensearch_client, VECTOR_FIELD_NAME 

K_VALUE = 5 

def improved_hybrid_retriever(query: str, client: OpenSearch, embeddings: GoogleGenerativeAIEmbeddings, k: int = K_VALUE) -> List[Document]:
    
    print("  - Starting improved hybrid search...")
    
    try:
        # 1. Generate Query Vector (Semantic Search Input)
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        print(f"Error generating embedding vector: {e}")
        return []

    # 2. Structured Filter Logic 
    lucene_filters = []
    lower_query = query.lower()

    # Query: Companies that are not having any risks (filters by total_findings: 0)
    if "no risk" in lower_query or "no finding" in lower_query or "zero risk" in lower_query:
        # Targeting the explicit 'total_findings' integer field
        lucene_filters.append({"term": {"metadata.doc_type": "ProfileSummary"}})
        lucene_filters.append({"term": {"metadata.total_findings": 0}}) 

    # Query: High priority risk (filters by Priority >= 2)
    elif "high priority" in lower_query or "priority 2" in lower_query or "critical risk" in lower_query:
         # Targeting the explicit 'priority' integer field
         lucene_filters.append({"range": {"metadata.priority": {"gte": 2, "lte": 4}}}) 
         lucene_filters.append({"term": {"metadata.doc_type": "RiskFinding"}})

    # Query: Drill downs for specific categories/names
    elif "h1b" in lower_query or "visa" in lower_query or "espionage" in lower_query:
        # Using the cleaned subcategory name from the corrected ingestion script
        lucene_filters.append({"term": {"metadata.risk_subcategory": "H1B visa and green card sponsorship"}})
        lucene_filters.append({"term": {"metadata.doc_type": "RiskFinding"}})

    elif "russian" in lower_query or "adversarial supply chain" in lower_query:
        # Targeting the specific subcategory
        lucene_filters.append({"term": {"metadata.risk_subcategory": "ADVERSARIAL SUPPLY CHAIN: Vends to an adversarial government"}})
        lucene_filters.append({"term": {"metadata.doc_type": "RiskFinding"}})
        
    elif "hong kong" in lower_query or "foci" in lower_query:
        # Targeting the specific category
        lucene_filters.append({"term": {"metadata.risk_category": "Foreign Ownership, Control, or Influence (FOCI)"}})
        lucene_filters.append({"term": {"metadata.doc_type": "RiskFinding"}})
        
    # Default filters: if no special filters, search all risk and summary docs
    if not lucene_filters:
        print("  - Applying default filter for all Risk Findings and Summaries.")
        lucene_filters.append(
            {"bool": {"should": [
                {"term": {"metadata.doc_type": "RiskFinding"}},
                {"term": {"metadata.doc_type": "ProfileSummary"}}
            ], "minimum_should_match": 1}}
        )

    # 3. Build the Hybrid Query Body
    query_body = {
        "query": {
            "bool": {
                "filter": lucene_filters, 
                "must": [
                    {
                        "knn": {
                            VECTOR_FIELD_NAME: {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    # Add a strong Lucene search to catch keyword matches where vector search might fail
                    {
                        "query_string": {
                            "query": query,
                            "fields": ["page_content", "metadata.company_name", "metadata.risk_category", "metadata.risk_subcategory"],
                            "default_operator": "OR" # Changed to OR for broader match
                        }
                    }
                ]
            }
        },
        "_source": ["page_content", "metadata"]
    }
    
    # 4. Execute the Search and Format Results
    try:
        response = client.search(index=INDEX_NAME, body=query_body)
        
        docs = []
        for hit in response['hits']['hits']:
            content = hit['_source']['page_content']
            metadata = hit['_source']['metadata']
            docs.append(Document(page_content=content, metadata=metadata))
        
        print(f"  - Retrieved {len(docs)} documents using Hybrid Search (Applied filters: {len(lucene_filters)}).")
        return docs
        
    except opensearch_exceptions.NotFoundError:
        print(f"  - Index {INDEX_NAME} not found during hybrid search. Did you run ingest.py?")
        return []
    except Exception as e:
        print(f"  - Error during hybrid OpenSearch query: {e}")
        # Print the problematic query body for external debugging
        print(f"    - Failing Query Body: {query_body}")
        return []

# --- RAG SIMULATION LOGIC (No changes needed here) ---

def run_rag_query(query: str, client: OpenSearch, embeddings: GoogleGenerativeAIEmbeddings):
    
    # 1. Retrieval
    retrieved_docs = improved_hybrid_retriever(query, client, embeddings) 
    
    if not retrieved_docs:
        print("\n[AI ASSISTANT RESPONSE]\nNo relevant context was retrieved from OpenSearch.")
        return

    # 2. Context Stuffing
    context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Generation (LCEL style prompt creation)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)

    SYSTEM_TEMPLATE = """
    You are a highly analytical risk assessment AI. Your role is to answer the user's question 
    **based ONLY on the provided CONTEXT**. Do not use external knowledge.
    
    - For high-priority risks, mention the priority level (e.g., Priority 2).
    - If a company has 'no detailed risk findings', explicitly state this.
    - Answer concisely and professionally.

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
        priority = doc.metadata.get('priority', 'N/A')
        doc_type = doc.metadata.get('doc_type', 'N/A')
        print(f"[{i+1}] {company_name} | Priority: {priority} | Doc Type: {doc_type}")
        print(f"     Snippet: {doc.page_content[:100]}...")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize shared components once
    os_client = get_opensearch_client()
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # The queries should now work better due to: 
    # 1. Corrected filtering logic using ranges/specific terms.
    # 2. The addition of a Lucene query (`query_string`) in the 'must' clause to boost keyword matches.
    test_queries = [
        "What are the risks associated with a company?", 
        "Which companies are having the high priority risk?", 
        "The companies that are not having any risks.", 
        "Drill down to the risk involving H1B visa sponsorship.", 
        "Find the relevant docs for the risk 'Adversarial Supply Chain' for MYNTRA.",
    ]
    
    for q in test_queries:
        run_rag_query(q, os_client, embeddings)