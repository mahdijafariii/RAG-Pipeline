import uvicorn
from fastapi import FastAPI
from retrieval import Retrieval
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from generation.generation import Generation
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
retriever = Retrieval(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    documents_address="./data",
    vector_db_address="./vector_db"
)

generator = Generation("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    results = retriever(request.query, n_returned_docs=request.top_k)
    return {"chunks": results}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
@app.post("/generate")
def generate(request: QueryRequest):
    chunks = retriever(request.query, n_returned_docs=5)
    filtered_chunks = [chunk for chunk in chunks["results"] if chunk["score"] > 0.5]

    context_text = " ".join([chunk["content"] for chunk in filtered_chunks])

    response = generator(prompt=f"Context: {context_text}\nQuestion: {chunks['query']}")

    return {"query": request.query, "response": response}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)