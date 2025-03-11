from fastapi import FastAPI
import chromadb

app = FastAPI()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("news_articles")


@app.get("/")
def home():
    return {"message": "ChromaDB API is running!"}


@app.post("/add")
def add_document(doc_id: str, text: str):
    collection.add(ids=[doc_id], documents=[text])
    return {"message": f"Added document {doc_id}"}


@app.get("/query")
def query_documents(query_text: str):
    results = collection.query(query_texts=[query_text], n_results=3)
    return results