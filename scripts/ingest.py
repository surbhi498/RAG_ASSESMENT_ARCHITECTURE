import os
import sys
import json
import requests
import uuid
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"BASE_DIR: {BASE_DIR}")
dotenv_path = os.path.join(BASE_DIR, ".env")
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)

EMBEDDING_API = os.getenv("EMBEDDING_API")
print(f"DEBUG - EMBEDDING_API: {EMBEDDING_API}")
if not EMBEDDING_API:
    raise ValueError("EMBEDDING_API environment variable is not set!")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def chunk_text(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def ingest_txt(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": f"{os.path.basename(filepath)}_{i}_{uuid.uuid4()}",
            "text": chunk,
            "source": "file"
        })
    return docs

def ingest_json(filepath):
    with open(filepath, "r") as f:
        items = json.load(f)
    docs = []
    for item in items:
        text = item.get("text", "")
        source = item.get("source", "json")
        base_id = item.get("id", str(uuid.uuid4()))
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{base_id}_{i}",
                "text": chunk,
                "source": source
            })
    return docs

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file.txt|file.json>")
        return
    filepath = sys.argv[1]
    if filepath.endswith(".json"):
        docs = ingest_json(filepath)
    else:
        docs = ingest_txt(filepath)

    payload = {"docs": docs}
    print(f"DEBUG - Sending payload to: {EMBEDDING_API}")
    resp = requests.post(EMBEDDING_API, json=payload)
    print(resp.status_code, resp.json())

if __name__ == "__main__":
    main()
