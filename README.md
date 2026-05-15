# Sherlock

A RAG-powered assistant for detectives. Upload case files (PDF or text) and ask questions about them. Answers are strictly grounded in the uploaded documents — if the answer isn't there, Sherlock says so.

## How it works

1. You upload a case file via the UI or API
2. The file gets chunked and stored in a local vector database (ChromaDB)
3. When you ask a question, the top matching chunks are retrieved and passed to Gemini with a strict prompt that forbids guessing
4. If the answer isn't in the evidence, the model returns: *"I don't have enough evidence to answer that."*

## Stack

- **Backend**: FastAPI (Python)
- **RAG**: LangChain + ChromaDB
- **LLM + embeddings**: Google Gemini (`gemini-2.5-flash-lite`, `gemini-embedding-001`)
- **Frontend**: Single HTML page with Tailwind CSS (served by FastAPI)
- **Container**: Docker + docker compose

## Running it

You need a free Google AI Studio API key: https://aistudio.google.com/apikey

```bash
cp .env.example .env
# add your key to .env

docker compose up --build
```

Then open http://localhost:8000.

The vector store is saved to `./data/chroma_db` and mounted as a volume, so indexed files survive container restarts.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload a `.pdf` or `.txt` case file |
| POST | `/api/ask` | Ask a question — body: `{"question": "..."}` |
| GET | `/api/documents` | List indexed files |
| DELETE | `/api/documents` | Clear the knowledge base |

```bash
# upload
curl -F "file=@case.txt" http://localhost:8000/api/upload

# ask
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Mrs. Hudson alibi?"}'
```

## Tests

```bash
pip install -r requirements.txt

# fast, no API key needed
pytest tests/test_api.py

# real API smoke test (needs GOOGLE_API_KEY)
RUN_INTEGRATION=1 pytest tests/test_engine.py
```
