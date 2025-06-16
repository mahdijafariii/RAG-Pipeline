# RAG-Pipeline Backend

This is the backend of the RAG (Retrieval-Augmented Generation) pipeline, built using **FastAPI**. It combines information retrieval with large language models (LLMs) to provide high-quality, context-aware answers from documents or a custom knowledge base.

## 🔧 Features

- ✅ FastAPI-powered RESTful API
- 📄 Document chunking and semantic indexing
- 🔍 Retrieval system for fetching relevant context
- 🤖 Pluggable LLM interfaces (OpenAI, Hugging Face, Gemini, etc.)
- 🔁 Modular architecture with interchangeable components
- 🧪 Easily testable and extendable

---

## 🧱 Project Structure

RAG-Pipeline/
├── generation/
│ ├── init.py
│ ├── generation.py # Main generation manager
│ ├── huggingface.py # Hugging Face LLM integration
│ ├── gemeni.py # Gemini (Google) LLM integration
│ ├── together.py # Together.ai API wrapper
│ ├── ollama.py # Local LLM support via Ollama
├── retrieval.py # Context retrieval from indexed documents
├── main.py # FastAPI app entry point
├── requirements.txt # Project dependencies


---

## 🧠 How it Works

1. **Document Embedding**: Input documents are chunked and vectorized using embedding models.
2. **Context Retrieval**: At query time, relevant chunks are retrieved using similarity search.
3. **Generation**: A selected LLM is used to generate an answer based on the retrieved context.
4. **Response**: The system returns the context-aware answer to the client.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/RAG-Pipeline.git
cd RAG-Pipeline
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

✅ Supported LLM Engines
- OpenAI (gpt-3.5, gpt-4)
- HuggingFace models (e.g., bert-base, flan-t5)
- Google Gemini
- Together AI
- Ollama (local deployment)


