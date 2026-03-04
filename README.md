Financial Document RAG Pipeline
A privacy-first Retrieval-Augmented Generation (RAG) system built for financial document analysis, featuring a REST API, Streamlit UI, and 100% local inference capabilities.

What It Does
This system allows users to upload dense financial PDFs (like 10-K reports or earnings transcripts), ask questions in natural language, and receive precise answers with exact source citations. It is designed from the ground up to support highly sensitive financial data by enabling completely local, offline execution using local embedding models and Ollama, while retaining the flexibility to route to cloud providers like Claude and GPT when needed.

Why I Built This Project
I developed this pipeline to solve a specific problem in financial machine learning: the need for absolute data privacy without sacrificing the power of modern LLMs. I wanted an architecture that:

Ensures zero data leakage by running embeddings and LLM generation 100% locally.
Avoids vendor lock-in by supporting multiple LLM providers (Ollama, Claude, GPT) via a clean abstraction layer.
Relies on minimal dependencies (utilizing specific utility packages rather than monolithic frameworks).
Functions seamlessly as both an importable library and a scalable REST API.
The primary focus is on engineering clarity, privacy, and architectural extensibility.

Key Features
100% Local Execution Option: Full support for local models via Ollama, ensuring zero API costs and maximum data privacy.
Multi-Provider LLM Support: Easily switch between Ollama, Claude, and GPT.
Robust Document Processing: Uses pypdf with a pdfplumber fallback for complex financial layouts and tables.
Local Embeddings: SentenceTransformers for free, private, and fast vectorization.
Efficient Vector Search: In-memory FAISS vector store for sub-millisecond semantic search.
Streaming Responses: Real-time generation via Server-Sent Events (SSE).
Source Citations: Transparent page number and context tracking.
Interactive UI: FastAPI backend paired with a Streamlit frontend demonstration.

Architecture Choices
Why FAISS?
For this specific implementation (handling <1000 documents per session), FAISS provides exact similarity search with minimal infrastructure overhead. It runs entirely in-memory, requiring only ~100MB of RAM per 10,000 documents (using 384-dimensional embeddings). For multi-tenant scaling, the data access layer is designed to easily migrate to Qdrant or Pinecone.

Why Local Embeddings & Ollama?
Financial data is highly proprietary. By utilizing SentenceTransformers (all-MiniLM-L6-v2) and local LLMs via Ollama, the system guarantees:

Zero Cost Inference: No API billing for document ingestion or querying.
Data Sovereignty: Highly sensitive financial numbers never leave the local machine or internal server.

System Flow
┌─────────────┐
│ PDF Upload  │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│ Document Processor  │ ← pypdf / pdfplumber
└──────┬──────────────┘
       │
       ↓
┌─────────────────┐
│ Text Chunking   │ ← RecursiveCharacterTextSplitter
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Embeddings      │ ← SentenceTransformers (local)
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│ Vector Store    │ ← FAISS IndexFlatIP
└──────┬──────────┘
       │
       ↓
┌─────────────────┐    ┌──────────────┐
│ User Query      │───→│ Embedding    │
└─────────────────┘    └──────┬───────┘
                              │
                              ↓
                       ┌──────────────┐
                       │ Similarity   │
                       │ Search (k=5) │
                       └──────┬───────┘
                              │
                              ↓
                       ┌──────────────┐
                       │ LLM Provider │ ← Ollama / Claude / GPT
                       │ Generation   │
                       └──────┬───────┘
                              │
                              ↓
                       ┌──────────────┐
                       │ Answer +     │
                       │ Citations    │
                       └──────────────┘

Note: The /query/stream endpoint emits Server-Sent Events (sources → token chunks → usage → done) for real-time UI updates.

Quick Start
Installation
# Clone repository
git clone https://github.com/yourusername/financial-rag.git
cd financial-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

Configuration & Setup
Start your local Ollama instance (ensure you have pulled your preferred model, e.g., ollama run llama3).

# Optional: Set cloud API keys if not using Ollama exclusively
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'

Run the Application
# Start the FastAPI Backend
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# In a new terminal, start the Streamlit UI
streamlit run app.py

API Documentation will be instantly available at: http://localhost:8000/docs

Usage Examples
Python Client (Local Model Setup)
from rag.complete_rag_system import CompleteRAGSystem
from rag.providers.ollama import OllamaProvider

# Initialize with zero-cost local provider
provider = OllamaProvider(model="llama3", host="http://localhost:11434")
rag = CompleteRAGSystem(llm_provider=provider)

# Index financial documents
rag.index_documents([
    "docs/tsla_q4_earnings.pdf",
    "docs/aapl_annual_report.pdf"
])

# Query the system
response = rag.query(
    "What were the primary supply chain risks mentioned in Q4?",
    k=5,
    min_score=0.5
)

print(f"Analysis: {response.answer}")
print(f"Sources utilized: {len(response.sources)}")

Extending Providers
The abstraction layer makes it trivial to swap logic. For example, to switch from local execution to cloud:

from rag.providers.anthropic import AnthropicProvider
provider = AnthropicProvider(api_key="your-key", model="claude-3-sonnet")

Tech Stack
LLM Engine: Ollama (Local), Anthropic Claude, OpenAI GPT

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

Vector Database: FAISS (IndexFlatIP)

Document Processing: pypdf, pdfplumber

Backend Framework: FastAPI (Async)

Frontend / Demo: Streamlit

Evaluation
To test retrieval quality and generation accuracy on your specific financial datasets:

python examples/evaluate_rag.py

Author
Vaibhav Patil
Contact: vaibhavofficial413@gmail.com

Built for robust, private financial document analysis.