# Financial Document RAG

RAG (Retrieval Augmented Generation) system for financial document analysis with REST API and Streamlit UI.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What It Does

Upload financial PDFs, ask questions in natural language, and get answers with source citations. The system tracks API costs and supports local Ollama models, Claude, and GPT.

## Why This Project?

I built this while learning about RAG systems and wanted something that:
- Supports multiple LLM providers (Ollama, Claude, GPT) without vendor lock-in
- Uses minimal dependencies (no full LangChain, just the text splitter)
- Has clean separation between RAG logic and API layer
- Can be used both as a library and as a REST API

The focus is on clarity and extensibility rather than covering every edge case.

## Features

- Multi-provider LLM support (Ollama locally, Claude, GPT) with easy switching
- Document processing: pypdf with pdfplumber fallback for complex layouts
- Local embeddings via SentenceTransformers (free, private)
- FAISS vector store for semantic search
- Streaming responses via Server-Sent Events
- Source citations with page numbers
- Cost tracking for API usage
- FastAPI server + Streamlit demo UI

## Architecture Choices

### Why FAISS?

For this project (single-user, <1000 docs), FAISS gives us exact similarity search with minimal setup. It runs in-memory and is fast enough for most use cases. If you need multi-tenancy or millions of documents, migrating to Qdrant or Pinecone is straightforward.

RAM usage: ~100MB per 10k documents (384-dim embeddings).

### Why LangChain's Text Splitter?

Rather than write a custom chunker, I use `langchain-text-splitters` (just the splitter package, not the full framework). It's mature, tested, and handles edge cases I'd probably miss.

### Why Local Embeddings?

SentenceTransformers (all-MiniLM-L6-v2) runs locally, which means:
- Zero cost
- No data leaves your machine
- Fast inference (~50ms per query)

The quality is good for most document types. If you need higher accuracy, switching to OpenAI embeddings is a simple config change.

## Architecture

### System Architecture

```
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
                       │ LLM Provider │ ← OLLMA/Claude / GPT
                       │ Generation   │
                       └──────┬───────┘
                              │
                              ↓
                       ┌──────────────┐
                       │ Answer +     │
                       │ Citations    │
                       └──────────────┘
```

**Note**: The `/query/stream` endpoint follows the same flow but emits Server-Sent Events (sources → token chunks → usage → done) instead of a single JSON response.

### Code Architecture

```
project/
├── rag/                        # Core RAG system
│   ├── document_processor.py   # PDF loading & chunking
│   ├── embedding_and_vectorstore.py  # Embeddings & FAISS
│   ├── rag_generator.py        # Answer generation
│   ├── complete_rag_system.py  # End-to-end pipeline
│   └── providers/              # LLM providers (abstraction)
│       ├── base.py
│       ├── anthropic.py
│       └── openai.py
│
├── api/                        # FastAPI REST API
│   ├── main.py                 # App entry point
│   ├── config.py               # Configuration
│   ├── dependencies.py         # Dependency injection
│   ├── exceptions.py           # Custom exceptions
│   ├── controllers/            # HTTP layer (thin)
│   │   ├── health.py
│   │   ├── system.py
│   │   ├── documents.py
│   │   └── query.py
│   ├── services/               # Business logic
│   │   ├── rag_service.py
│   │   └── document_service.py
│   └── models/                 # Request/Response models
│       ├── requests.py
│       └── responses.py
│
├── app.py                      # Streamlit UI (demo)
├── examples/                   # Usage examples
│   ├── ....
│   └── evaluate_rag.py         # Evaluation framework
└── requirements.txt
```

Controllers are thin HTTP handlers, business logic lives in services. This makes testing easier and lets you swap LLM providers by changing one line in the config.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/financial-rag.git
cd financial-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

For local development with zero cost, simply download and start Ollama (e.g. `ollama run llama3`). 

If you wish to use cloud models, configure your API keys:

```bash
# Set API key (choose one)
export ANTHROPIC_API_KEY='your-anthropic-key'
# or
export OPENAI_API_KEY='your-openai-key'

# Optional: create .env file
echo "ANTHROPIC_API_KEY=your-key" > .env
```

### Run API Server

```bash
# Development mode
cd api
python main.py

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at:
- Swagger docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/api/health

### Run Streamlit Demo

```bash
# Quick demo interface
streamlit run app.py
```

## API Usage

### Using Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Initialize the RAG system
response = requests.post(f"{BASE_URL}/system/initialize", json={
    "provider": "anthropic",
    "api_key": "your-key",
    "model": "claude-sonnet-4-20250514"
})

# Upload documents
files = [
    ("files", ("report_q4.pdf", open("report_q4.pdf", "rb"), "application/pdf")),
    ("files", ("annual_report.pdf", open("annual_report.pdf", "rb"), "application/pdf"))
]
response = requests.post(f"{BASE_URL}/documents/upload", files=files)

# Ask a question
response = requests.post(f"{BASE_URL}/query", json={
    "question": "Qual è stata la crescita dei ricavi nel Q4 2023?",
    "k": 5,
    "min_score": 0.5
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Cost: ${result['cost_usd']:.6f}")
print(f"Sources: {len(result['sources'])}")
```

### Using curl

```bash
# Initialize
curl -X POST "http://localhost:8000/api/system/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "anthropic",
    "api_key": "your-key",
    "model": "claude-sonnet-4-20250514"
  }'

# Upload documents
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "files=@report_q4.pdf" \
  -F "files=@annual_report.pdf"

# Query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the Q4 revenue growth?",
    "k": 5,
    "min_score": 0.5
  }'
```

### Streaming Responses

The API supports real-time streaming for immediate feedback during answer generation.

#### Using Python with SSE Client

```python
import requests
import json

BASE_URL = "http://localhost:8000/api"

# Query with streaming
response = requests.post(
    f"{BASE_URL}/query/stream",
    json={
        "question": "Qual è stata la crescita dei ricavi nel Q4 2023?",
        "k": 5,
        "min_score": 0.5
    },
    stream=True
)

# Process SSE events
for line in response.iter_lines():
    if line:
        # Remove "data: " prefix
        data = line.decode('utf-8').replace('data: ', '')
        event = json.loads(data)

        if event['type'] == 'sources':
            print(f"Retrieved {len(event['documents'])} documents")

        elif event['type'] == 'token':
            print(event['text'], end='', flush=True)

        elif event['type'] == 'usage':
            print(f"\n\nCost: ${event['cost_usd']:.6f}")
            print(f"Tokens: {event['total_tokens']}")

        elif event['type'] == 'done':
            print("\nStream completed")
            break

        elif event['type'] == 'error':
            print(f"Error: {event['message']}")
            break
```

#### Using curl with SSE

```bash
curl -N -X POST "http://localhost:8000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the Q4 revenue growth?",
    "k": 5,
    "min_score": 0.5
  }'
```

#### Event Types

The stream emits Server-Sent Events (SSE) with the following types:

| Event Type | Description                            | Payload                                     |
|------------|----------------------------------------|---------------------------------------------|
| `sources`  | Retrieved documents from vector search | `{"type": "sources", "documents": [...]}`   |
| `token`    | Text chunk as it's generated           | `{"type": "token", "text": "..."}`          |
| `usage`    | Token usage and cost metadata          | `{"type": "usage", "cost_usd": 0.002, ...}` |
| `done`     | Stream completed successfully          | `{"type": "done"}`                          |
| `error`    | Error occurred during processing       | `{"type": "error", "message": "..."}`       |

Streaming gives you real-time responses instead of waiting for the full answer. Useful for long documents and building responsive UIs.

## Direct Python Usage

```python
from rag.complete_rag_system import CompleteRAGSystem
from rag.providers.ollama import OllamaProvider

# Initialize with zero-cost local provider
provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
rag = CompleteRAGSystem(llm_provider=provider)

# Index documents
rag.index_documents([
    "docs/report_q4_2023.pdf",
    "docs/annual_report_2023.pdf"
])

# Query
response = rag.query(
    "What was the revenue growth in Q4?",
    k=5,
    min_score=0.5
)

print(response.answer)
print(f"Cost: ${response.cost_usd:.6f}")

# Save for later
rag.save("my_rag_index")

# Load existing index
rag = CompleteRAGSystem.load("my_rag_index", llm_provider=provider)
```

## Tech Stack

- LLMs: Ollama (Local), Anthropic Claude 4, OpenAI GPT-4o
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Vector Store: FAISS IndexFlatIP
- Document Processing: pypdf + pdfplumber fallback
- API: FastAPI with async support
- UI: Streamlit

## Evaluation

Evaluate system quality on test queries:

```bash
python examples/evaluate_rag.py
```

Tracks: retrieval score, precision@k, answer length, token usage, and citation quality.

## Configuration

### Environment Variables

```bash
# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# API Server
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# RAG Parameters
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
DEFAULT_K_RESULTS=5
DEFAULT_MIN_SCORE=0.5
```

### Switching LLM Providers

```python
# Use Claude
from rag.providers import AnthropicProvider
provider = AnthropicProvider(api_key="your-key", model="claude-sonnet-4-20250514")

# Or use GPT
from rag.providers import OpenAIProvider
provider = OpenAIProvider(api_key="your-key", model="gpt-4o-mini")
```

## Author

**Vaibhav Patil**

- LinkedIn: [Profile](https://www.linkedin.com/in/vaibhav-patil225/)
- GitHub: [@vaibuzz](https://github.com/vaibuzz/)

---

Built for robust, private financial document analysis.
Built with Ollama, Claude, GPT, FAISS, and SentenceTransformers.

This project is for learning/portfolio purposes. If you deploy it in production, add proper authentication and rate limiting.
