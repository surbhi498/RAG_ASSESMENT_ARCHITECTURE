from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
import time
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

print("DEBUG - MONGO_URI:", os.getenv("MONGO_URI"))

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set!")

# Initialize MongoDB client
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
try:
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

DB_NAME = "rag_db"
COLLECTION_NAME = "docs"
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Initialize SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

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
                "index": "vector_index"
            }
        },
        {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]

    results = list(collection.aggregate(pipeline))
    return {"results": results}
