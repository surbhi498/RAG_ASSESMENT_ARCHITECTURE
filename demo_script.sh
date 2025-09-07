#!/bin/bash

# Base URLs
EMBEDDING_URL="http://0.0.0.0:8001"
ORCHESTRATOR_URL="http://127.0.0.1:3000"
SEARCH_URL="http://0.0.0.0:8001/search"

echo "=== Step 1: Ingesting a document ==="

# Example document to embed
read -r -d '' DOC_PAYLOAD << EOM
{
  "id": "doc_1",
  "text": "Retrieval-Augmented Generation (RAG) combines traditional search with generative models."
}
EOM

curl -X POST "${EMBEDDING_URL}/embed" \
  -H "Content-Type: application/json" \
  -d "${DOC_PAYLOAD}"

echo -e "\nDocument ingested.\n"

echo "=== Step 2: Asking a context-aware query ==="

read -r -d '' QUERY_PAYLOAD << EOM
{
  "user_id": "test_user",
  "query": "What is RAG?",
  "k": 5
}
EOM

curl -X POST "${ORCHESTRATOR_URL}/chat" \
  -H "Content-Type: application/json" \
  -d "${QUERY_PAYLOAD}"

echo -e "\nQuery complete.\n"

echo "=== Step 3: Loading and applying a saved LoRA adapter for inference ==="
python3 ./scripts/lora_inference.py



