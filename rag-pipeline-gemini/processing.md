Phase 1: Data Ingestion

    This is the ingest.py script's job:

    Load Source Doc: The script starts by loading your complex data/knowledge-base.json file. This source document can be fed from the Opensearch itself/ArangoDB. For POC, we used local file

    Process & Chunk: Instead of splitting, your process_risk_data_to_documents function logically flattens the JSON, creating one Document for each risk finding and one for the summary.

    Embed & Store: LangChain's OpenSearchVectorSearch.from_documents function:

        Takes the page_content of each Document and sends it to GoogleGenAI Embeddings to create a vector.

        Stores that vector, along with its corresponding metadata, in the Opensearch Vector Store.

Phase 2: RAG Pipeline

    This is the query.py script's job:

        User Query: A user asks a question in plain text.

        Embed Query: The retriever automatically sends this text query to the GoogleGenAI Embeddings model to create a matching query vector.

        Search: This vector is sent to the Opensearch Vector Store, which performs a similarity search.

        Retrieve Context: Opensearch returns the "Top-K" (e.g., k=5) most relevant Document chunks (the original text and metadata) that match the query.

        Augment Prompt: LangChain (using LCEL) takes the retrieved documents (Context) and the original query and builds the final prompt for the LLM.

        Generate: The augmented prompt is sent to the Google Gemini Pro LLM, which synthesizes the context to generate a direct answer.

        Answer: The user receives the final answer.

LCEL stands for LangChain Expression Language.