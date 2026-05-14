def _upload(client, name, body):
    return client.post("/api/upload", files={"file": (name, body.encode(), "text/plain")})


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_no_docs_on_startup(client):
    r = client.get("/api/documents")
    assert r.json() == {"documents": []}


def test_ask_no_docs(client):
    r = client.post("/api/ask", json={"question": "Who did it?"})
    assert "enough evidence" in r.json()["answer"].lower()


def test_upload_and_ask(client):
    _upload(client, "case.txt", "The butler was in the garden.")

    assert client.get("/api/documents").json()["documents"] == ["case.txt"]

    r = client.post("/api/ask", json={"question": "Where was the butler?"})
    assert r.json()["sources"][0]["source"] == "case.txt"


def test_clear(client):
    _upload(client, "case.txt", "evidence")
    client.delete("/api/documents")
    assert client.get("/api/documents").json()["documents"] == []


def test_bad_file_extension(client):
    r = client.post("/api/upload", files={"file": ("case.png", b"\x89PNG", "image/png")})
    assert r.status_code == 400


def test_frontend_loads(client):
    r = client.get("/")
    assert "Sherlock" in r.text
