import uvicorn
from fastapi import FastAPI, Body
from retrieval import Retrieval
from pydantic import BaseModel

app = FastAPI()

retriever = Retrieval(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    documents_address="./data",
    vector_db_address="./vector_db"
)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    results = retriever(request.query, n_returned_docs=request.top_k)
    return {"chunks": results}

# @app.post("/generate")
# def generate(request: QueryRequest):
#     try:
#         # TODO: Instantiate and run the generator
#         response = "This is a test response"
#         return {"query": request.query, "response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)