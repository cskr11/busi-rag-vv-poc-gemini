
Example:

{
  "_id": "risk_doc_421a",
  "_index": "risk_knowledge_base", 
  "_score": 1.0, 
  "_source": {
    "page_content": "Risk Finding: The customer exhibited 5 instances of cross-border transfers to a high-risk jurisdiction (Country X) in the last quarter, totaling $850,000.\nCategory: AML (Subcategory: Geographic Risk)",
    "metadata": {
      "business_id": "ABC-987",
      "business_address": "123 Main St, Anytown, USA",
      "doc_type": "RiskFinding",
      "risk_category": "AML",
      "priority": "High",
      "source": "Transaction Monitoring System"
    },
    "vector_field": [
      0.015,
      -0.038,
      0.112,
      ...,
      -0.009,
      0.045,
      -0.071
    ]
  }
}

That's a great way to tie it all together! Here is an example of a single document record in its raw format, showing the Vector, Chunk, and Metadata as it would be indexed in OpenSearch.

We'll use a simplified version of your risk data structure.

💾 Raw Document Example in OpenSearch
This example shows the JSON document that would be indexed into your OpenSearch cluster under your INDEX_NAME.

JSON

{
  "_id": "risk_doc_421a",
  "_index": "risk_knowledge_base", 
  "_score": 1.0, 
  "_source": {
    "page_content": "Risk Finding: The customer exhibited 5 instances of cross-border transfers to a high-risk jurisdiction (Country X) in the last quarter, totaling $850,000.\nCategory: AML (Subcategory: Geographic Risk)",
    "metadata": {
      "business_id": "ABC-987",
      "business_address": "123 Main St, Anytown, USA",
      "doc_type": "RiskFinding",
      "risk_category": "AML",
      "priority": "High",
      "source": "Transaction Monitoring System"
    },
    "vector_field": [
      0.015,
      -0.038,
      0.112,
      ...,
      -0.009,
      0.045,
      -0.071
    ]
  }
}
🔬 What Each Component Conveys
1. The Chunk (page_content)
This is the raw text that the AI model will read and summarize. It conveys the specific, human-readable details of the risk.

Format: A simple string of text.

Conveys: The specific facts about the risk event (e.g., the number of transfers, the amount, the countries involved). This is what the system returns when it finds a match.

2. The Vector (vector_field)
This is the numerical representation of the chunk's meaning. It's used exclusively by the OpenSearch k-NN engine for searching.

Format: A large array (list) of floating-point numbers. (e.g., 768 numbers long).

Conveys: The semantic concept of the chunk. In this case, this specific vector represents the meaning of "high-value cross-border financial activity associated with geographic risk." Any query that asks about this concept will have a similar vector and be retrieved.

3. The Metadata (metadata)
This is structured, indexed information used to refine the search. It conveys necessary context for filtering and post-processing.

Format: A JSON object of key-value pairs.

Conveys: Context and Search Filters.

You can search for the vector only if the priority is 'High'.

The LLM knows the document is a RiskFinding related to AML for business ABC-987.

These fields are searchable using standard OpenSearch term or range queries, letting you combine filter searches with vector similarity searches.