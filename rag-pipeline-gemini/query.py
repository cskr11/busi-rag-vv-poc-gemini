# query.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OpenSearchVectorSearch

# Core components for the modern LCEL RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, LLM_MODEL

# Helper function to format retrieved documents into a single string
def format_docs(docs):
    """Concatenates document pages into a single string for the LLM context."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_pipeline(query: str):
    """
    Initializes the RAG components and runs a query using LCEL.
    """
    print(f"-> Initializing RAG pipeline...")

    # 1. Initialize Embeddings (Required to interact with the indexed vectors)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 2. Create the Vector Store Client (Retrieval Setup)
    try:
        vector_store = OpenSearchVectorSearch(
            index_name=INDEX_NAME,
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL,
            # MODIFICATION: Added client_kwargs for consistent connection
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        print(f"   - Connected to OpenSearch index: {INDEX_NAME}")
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return

    # 3. Initialize LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    print(f"   - Initialized LLM: {LLM_MODEL}")

    # 4. Define the RAG Prompt Template
    template = """
    You are an expert risk analyst. Use only the following context to answer the user's question.
    If you don't know the answer based on the context, state that you don't have enough information.
    Be concise and professional.

    Context:
    {context}
    """

    # Use ChatPromptTemplate to properly structure the prompt for the Chat LLM
    custom_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{question}")
        ]
    )
    print("   - Defined RAG Prompt Template.")

    # 5. Create the RAG Chain using LCEL
    # We will retrieve 5 documents to get a good amount of context
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # The LCEL RAG chain defines the sequence of operations:
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    # 6. Run the query
    print(f"\n-> Running Query: **{query}**")

    # Execute the chain
    final_answer = rag_chain.invoke(query)

    # Separately retrieve the source docs to display them
    # Use a try-except block in case the retriever finds no documents
    try:
        source_docs = retriever.invoke(query)
    except Exception as e:
        print(f"   - Error retrieving source documents: {e}")
        source_docs = []

    # 7. Output Results
    print("\n[AI Generated Answer]")
    print(final_answer)
    print("\n[Source Documents (Context Used)]")

    if not source_docs:
        print("   - No source documents were retrieved.")

    for i, doc in enumerate(source_docs):
        print(f"--- Source {i+1} (Type: {doc.metadata.get('doc_type', 'N/A')}) ---")
        print(f"Content Snippet: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")

    return final_answer

# --- MAIN EXECUTION (MODIFIED) ---

if __name__ == "__main__":
    # Ensure you run ingest.py successfully before running this.

    # MODIFICATION: Replaced the old query with relevant queries for your data.

    # Query 1: A broad, summary-level question
    print("--- Running Broad Query ---")
    test_query_1 = "What is the overall risk score for the business at 3039 East Cornwallis RD?"
    run_rag_pipeline(test_query_1)

    print("\n" + "="*50 + "\n")

    # Query 2: A specific, detailed question
    print("--- Running Specific Query ---")
    test_query_2 = "What are the high-priority legal or financial risks for this company?"
    run_rag_pipeline(test_query_2)