from fastapi.testclient import TestClient


def test_ask_without_documents_returns_safe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app.main import app

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the policy?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "enough information" in response.json()["answer"]

