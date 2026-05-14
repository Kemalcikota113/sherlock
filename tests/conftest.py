import os

# try to load the API key from .env so integration tests work without
# having to manually export the variable. unit tests use a dummy key anyway.
if "GOOGLE_API_KEY" not in os.environ and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.strip().split("=", 1)[1]
                break

os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key")

import pytest
from fastapi.testclient import TestClient

# A fake engine that doesn't do any real embedding or LLM calls, for use in unit tests.
class FakeEngine:

    def __init__(self):
        self._docs: dict[str, str] = {}

    def ingest(self, file_path: str, filename: str) -> int:
        with open(file_path, "rb") as f:
            self._docs[filename] = f.read().decode("utf-8", errors="ignore")
        return 1

    def query(self, question: str) -> dict:
        if not self._docs:
            return {"answer": "I don't have enough evidence to answer that.", "sources": []}
        filename, text = next(iter(self._docs.items()))
        return {
            "answer": f"Based on the evidence, the answer to '{question}' is in {filename}.",
            "sources": [{"source": filename, "page": 0, "snippet": text[:240]}],
        }

    def list_documents(self) -> list[str]:
        return sorted(self._docs.keys())

    def clear(self) -> None:
        self._docs.clear()


@pytest.fixture
def client():
    from app.main import app, get_engine

    fake = FakeEngine()
    app.dependency_overrides[get_engine] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
