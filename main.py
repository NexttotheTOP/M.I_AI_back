from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Ensure the ChromaDB folder exists
CHROMA_PATH = "./chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)  # Persistent storage
collection = chroma_client.get_or_create_collection("news_articles")

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY in Heroku Config Vars.")

openai.api_key = OPENAI_API_KEY

# Function to generate OpenAI embeddings
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]  # Correct format
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")

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
            embedding = get_embedding(text)
            collection.add(
                ids=[f"{request.ticker}-{i}"],
                embeddings=[embedding],
                documents=[text]
            )

        return {"message": f"Stored news data for {request.ticker} in ChromaDB"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing news: {str(e)}")

# Retrieve relevant news
@app.post("/retrieve_news")
def retrieve_news(request: QueryRequest):
    try:
        query_embedding = get_embedding(request.question)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        # Ensure results["documents"] exists and is not None
        documents = results.get("documents", [])
        if not documents:
            return {"documents": []}

        return {"documents": documents[0]}  # Returns the first list of results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving news: {str(e)}")

# Test Route
@app.get("/")
def home():
    return {"message": "ChromaDB Backend is running!"}