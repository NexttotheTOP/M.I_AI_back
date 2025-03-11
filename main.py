from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
collection = chroma_client.get_or_create_collection("news_articles")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define request models
class NewsStoreRequest(BaseModel):
    ticker: str
    newsSummary: str
    articles: list

class QueryRequest(BaseModel):
    question: str

# Store news in ChromaDB
@app.post("/store_news")
def store_news(request: NewsStoreRequest):
    try:
        all_texts = [request.newsSummary] + [article['description'] for article in request.articles]

        for i, text in enumerate(all_texts):
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                ids=[f"{request.ticker}-{i}"],
                embeddings=[embedding],
                documents=[text]
            )

        return {"message": f"Stored news data for {request.ticker} in ChromaDB"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieve relevant news
@app.post("/retrieve_news")
def retrieve_news(request: QueryRequest):
    try:
        query_embedding = embedding_model.encode(request.question).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        return {"documents": results["documents"][0]} if results["documents"] else {"documents": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test Route
@app.get("/")
def home():
    return {"message": "ChromaDB Backend is running!"}