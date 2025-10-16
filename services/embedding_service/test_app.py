from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
import time

app = FastAPI()
# ----- Config -----
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag_db"
COLLECTION_NAME = "docs"

# Mongo connection
# ...existing code...
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    print(f"MongoDB connection error: {e}")
# # ...existing code...
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# Ensure vector index exists in Atlas manually or via Atlas UI
# Example: create index on "embedding" field with type "vector" (cosine)

# SentenceTransformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app


# ----- Models -----
class Doc(BaseModel):
    id: str
    text: str
    source: str = "manual"

class BulkDocs(BaseModel):
    docs: List[Doc]

class SearchRequest(BaseModel):
    query: str
    k: int = 5

# ----- Helpers -----
def embed_text(text: str):
    vec = embedder.encode(text)
    return vec.tolist()

# ----- Endpoints -----
@app.post("/embed")
def embed(doc: Doc):
    vector = embed_text(doc.text)
    return {"id": doc.id, "embedding": vector, "dim": len(vector)}

@app.post("/bulk_embed")
def bulk_embed(payload: BulkDocs):
    docs = []
    for doc in payload.docs:
        vector = embed_text(doc.text)
        docs.append({
            "_id": doc.id,
            "text": doc.text,
            "source": doc.source,
            "embedding": vector,
            "created_at": time.time()
        })
    for d in docs:
        collection.update_one({"_id": d["_id"]}, {"$set": d}, upsert=True)
    return {"upserted": len(docs)}

@app.post("/search")
def search(req: SearchRequest):
    vector = embed_text(req.query)
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": vector,
                "path": "embedding",
                "numCandidates": 100,
                "limit": req.k,
                "index": "vector_index"  # name of your Atlas search index
            }
        },
        {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results = list(collection.aggregate(pipeline))
    return {"results": results}
