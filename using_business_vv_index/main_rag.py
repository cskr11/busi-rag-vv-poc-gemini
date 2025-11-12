import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Define your LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "rag_knowledge_index"
LLM_MODEL = "gemini-2.5-pro" # Using the specified Pro model
EMBEDDING_MODEL = "models/text-embedding-004" 

def run_rag_pipeline(question: str):
    """Initializes LLM and Retriever, then runs the RAG chain."""
    
    # --- 1. Initialize LLM (Gemini Pro) ---
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1
    )

    # --- 2. Initialize Retriever (OpenSearch) ---
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    os_client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=('user', 'password') 
    )
    
    # Connect to the existing OpenSearch index
    vectorstore = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        http_auth=('user', 'password'), # Adjust auth
        client=os_client,
    )
    
    # Create the retriever instance (k=3 means retrieve 3 chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- 3. Define Prompt and Document Chain ---
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Q&A system. Use the following context to answer the question. 
        If you don't know the answer based on the context, state that you cannot find the answer.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """
    )

    # The document chain combines retrieved docs and the question into the LLM prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # --- 4. Create and Invoke the Full RAG Chain ---
    # The final chain: Retriever + Document Chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print(f"Asking: {question}\n")
    
    response = retrieval_chain.invoke({"input": question})
    
    print("--- GENERATED ANSWER ---")
    print(response["answer"])
    print("\n--- SOURCES (Context Retrieved) ---")
    for i, doc in enumerate(response["context"]):
        print(f"Source {i+1}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    user_question = "What was the main outcome of Project Alpha and how did it affect the Q4 financials?"
    run_rag_pipeline(user_question)