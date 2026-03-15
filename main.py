import os
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY in .env")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)

app = FastAPI(title="Linux Command RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


@app.get("/")
def home():
    return {"message": "Linux Command RAG API. Use /ask?q=your question"}


@app.get("/ask")
def ask(q: str = Query(..., description="Your question in natural language")):
    logging.info(f"Question: {q}")

    try:
        q_emb = embedder.encode(q).tolist()
        
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=1
        )

        if response and response.points and response.points[0].payload:
            cmd = response.points[0].payload.get("output", "")
            similar = [p.payload.get("output", "") for p in response.points[1:4] if p.payload]
            return {
                "question": q,
                "answer": cmd,
                "command": cmd,
                "similar_commands": similar
            }

        return {
            "question": q,
            "answer": "Command not found in database",
            "command": None,
            "similar_commands": []
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
