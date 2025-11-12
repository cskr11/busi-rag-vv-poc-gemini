# rag_pipeline.py

import os
# LLM and Embedding models
from langchain_google_genai import ChatGoogleGenerativeAI
# Core prompt components
from langchain_core.prompts import ChatPromptTemplate
# Core runnable component (used for the pipe operator)
from langchain_core.runnables import RunnablePassthrough
# Core document structure (already fixed in data_prep.py)
from langchain_core.documents import Document

# Import modular components
from data_prep import get_raw_weather_data, create_and_chunk_documents
from vector_store import get_opensearch_vector_store, index_documents, OPENSEARCH_VECTOR_INDEX
from dotenv import load_dotenv # Import the loader

# Load environment variables from .env file (for GOOGLE_API_KEY)
load_dotenv()

# --- Configuration ---
# The API key is now loaded from .env by the load_dotenv() call above.

# --- RAG Core Logic using LCEL pipe operator (|) ---

def create_rag_chain(llm, retriever):
    """
    Creates the Retrieval-Augmented Generation chain using LCEL pipe operators (Runnable).
    This avoids the unstable 'langchain.chains' imports.
    """

    # 1. Define the Prompt Template
    system_prompt = (
        "You are an expert risk assessment analyst. Answer the user's question based "
        "only on the provided context (weather data). "
        "If the context does not contain the answer, politely state that you cannot provide an answer. "
        "\n\nCONTEXT: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"), # Use 'input' for the user's query
        ]
    )

    # 2. Define the Document Formatting function
    # This prepares the list of documents retrieved by the retriever for the prompt's {context} variable.
    def format_docs(docs):
        # Join the page_content of all retrieved documents with a newline separator
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Define the LCEL Chain
    # The chain structure is:
    # {input, context} -> Prompt -> LLM (Gemini 2.5 Pro)
    rag_chain = (
        # 3a. Prepare the inputs:
        # 'context' comes from the retriever | 'input' is passed through from the main invoke call
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt # 3b. Inject inputs into the prompt
        | llm    # 3c. Pass the formatted prompt to the LLM
    )

    # We return the whole pipeline, including the retriever, to access source documents later
    return rag_chain

def run_rag_query(query: str):
    """
    Runs the full RAG pipeline.
    """

    # --- 1. Connect to Vector Store and Index Check ---
    vector_store = get_opensearch_vector_store()

    # Index Check (Logic remains the same)
    try:
        if not vector_store.client.indices.exists(index=OPENSEARCH_VECTOR_INDEX):
            print(f"Index '{OPENSEARCH_VECTOR_INDEX}' not found. Indexing data now...")
            # raw_data = get_raw_weather_data()
            chunks = create_and_chunk_documents()
            index_documents(vector_store, chunks)
        else:
            print(f"Index '{OPENSEARCH_VECTOR_INDEX}' found. Proceeding with retrieval.")

    except Exception as e:
        # Added a clearer instruction here
        print(f"Warning: Could not check/index data. Ensure OpenSearch is running and accessible.")
        print(f"Error: {e}")

    # Define the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # --- 2. Initialize LLM and RAG Chain ---
    # MODEL NAME ADJUSTED: gemini-pro -> gemini-2.5-pro
    print("\nInitializing Gemini 2.5 Pro LLM and RAG chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

    # Create the LCEL chain
    qa_chain = create_rag_chain(llm, retriever)

    # --- 3. Execute Query ---
    print(f"\n--- Running RAG Query: {query} ---")

    # The 'invoke' method for this LCEL chain uses the key 'input'
    result = qa_chain.invoke({"input": query})

    # --- 4. Output Results ---
    print("\n### RAG Final Answer (Gemini 2.5 Pro) ###")
    print(result.content) # LCEL LLM result is a BaseMessage object with content attribute

    print("\n### Source Documents (Context Retrieved from OpenSearch) ###")

    # To get source documents, we have to run the retriever separately or use a specialized LCEL component.
    # For a POC, running the retriever manually is the simplest way to display sources:
    source_documents = retriever.invoke(query)

    if source_documents:
        for i, source_doc in enumerate(source_documents):
            print(f"\n--- Source Document {i+1} ---")
            print(f"City: {source_doc.metadata.get('city', 'N/A')}")
            print(f"Condition: {source_doc.metadata.get('condition', 'N/A')}")
            print(f"Content Used: {source_doc.page_content}")
    else:
        print("No source documents were retrieved for this query.")

    return result

if __name__ == "__main__":
    try:
        query_to_run = "Which city is under a severe weather alert, and what are the details about wind speed and flights?"
        run_rag_query(query_to_run)
    except Exception as e:
        print(f"\n❌ A critical error occurred during the RAG pipeline execution.")
        print(f"Please ensure your OpenSearch container is running and accessible.")
        print(f"Error details: {e}")