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
    # The embedding function must match the one used during ingestion.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 2. Create the Vector Store Client (Retrieval Setup)
    try:
        vector_store = OpenSearchVectorSearch(
            index_name=INDEX_NAME,
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_URL
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
    You are an expert assistant. Use only the following context to answer the user's question.
    If you don't know the answer based on the context, state that you don't have enough information.

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
    print("- Defined RAG Prompt Template.")

    # 5. Create the RAG Chain using LCEL (This replaces RetrievalQA)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # The LCEL RAG chain defines the sequence of operations:
    rag_chain = (
        # 1. RunnablePassthrough() handles the question input.
        # It assigns two keys: 'context' (retrieved docs formatted) and 'question' (the original query).
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        # 2. Pass the dictionary to the prompt template
        | custom_prompt
        # 3. Invoke the LLM
        | llm
        # 4. Parse the final output to a string
        | StrOutputParser()
    )

    # 6. Run the query
    print(f"\n-> Running Query: **{query}**")

    # Execute the chain
    final_answer = rag_chain.invoke(query)

    # Separately retrieve the source docs to display them
    source_docs = retriever.invoke(query)

    # 7. Output Results
    print("\n[AI Generated Answer]")
    print(final_answer)
    print("\n[Source Documents (Context Used)]")
    for i, doc in enumerate(source_docs):
        print(f"--- Source {i+1} ---")
        # Ensure the content snippet is safe to print
        print(f"Content Snippet: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")

    return final_answer

if __name__ == "__main__":
    # Ensure you run ingest.py successfully before running this.
    test_query = "What is the typical height required for fall protection and what is OpenSearch used for?"
    run_rag_pipeline(test_query)