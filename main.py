from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow frontend domains
origins = [
    "http://localhost:5173",  # Local development
    "https://capable-bubblegum-2b89e8.netlify.app",
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Ensure the ChromaDB folder exists
CHROMA_PATH = "/tmp/chroma_db"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

# Initialize ChromaDB (Temporary Storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("news_articles")

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY in Heroku Config Vars.")


client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))



# Define request models
class NewsStoreRequest(BaseModel):
    ticker: str
    newsSummary: str
    articles: list

class QueryRequest(BaseModel):
    question: str

class SummarizeRequest(BaseModel):
    text: str
    ticker: str
    summaryType: str



# Function to generate OpenAI embeddings
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
    


@app.post("/summarize_news")
def summarize_news(request: SummarizeRequest):
    try:
        summary_instructions = {
            "concise": """Summarize the key financial insights in **2-3 sentences**.

            ### Format:
            - Use **bold** for key insights.
            - Keep the summary short and focused on critical financial impacts.
            """,

            "detailed": """Provide a well-structured, **balanced financial summary** with key market movements, economic impacts, and investor insights.

            ### Format:
            - **### Key Takeaways**
            - Use bullet points (- or *) for listing insights.
            - **bold** important market insights.
            """,

            "comprehensive": """Generate a **detailed** professional market analysis covering **key financial trends, economic drivers, and investment implications**.

            ### Format:
            - **### Key Takeaways**: Overview of the most crucial insights.
            - **### Market Impact**: Explain how these developments influence stock prices, economy, or investor behavior.
            - **### Expert Analysis**: Provide deeper insights into trends and investment strategies.
            - Use bullet points (- or *) for clarity.
            - Use **bold** for financial highlights.
            """
        }

        if request.summaryType not in summary_instructions:
            raise HTTPException(status_code=400, detail="Invalid summary type")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a highly skilled financial analyst with expertise in stock markets, cryptocurrencies, macroeconomics, and investment strategies. 
                    The user is requesting a **{request.summaryType.upper()}** market summary based on the articles provided specifically for **{request.ticker}**.  
                    Some articles provided may include irrelevant stocks or financial topics. 
                    **ONLY summarize news directly related to {request.ticker}** and ignore anything else.

                    {summary_instructions[request.summaryType]}

                    Now generate the summary in this exact Markdown format:"""
                },
                {
                    "role": "user",
                    "content": f"Summarize the financial news for **{request.ticker}** based on the following articles. Ignore anything unrelated:\n\n{request.text}"
                }
            ]
        )

        return {"summary": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing news: {str(e)}")


@app.post("/generate_answer")
def generate_answer(request: QueryRequest):
    try:
        # Retrieve relevant news from ChromaDB
        query_embedding = get_embedding(request.question)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        retrieved_docs = results.get("documents", [])

        if not retrieved_docs:
            return {"answer": "No relevant news found for this query."}

        # Generate AI response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial expert answering questions based on real news."},
                {"role": "user", "content": f"Here are the relevant news articles:\n\n{retrieved_docs}\n\nUser Question: {request.question}"}
            ]
        )

        return {"answer": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

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