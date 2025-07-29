import pytest
from fastapi.testclient import TestClient
from routes import create_router, session_cache
from fastapi import FastAPI

class FakeAppGraph:
    def invoke(self, state, config=None):
        return {
            "expected_format": "",
            "analysis_summary": "The user wants to count files.",
            "tool_history": [],
            "draft_solution": "Think: Use 'ls' and count.",
            "current_problem": "Count files in /etc.",
            "last_action": "```bash\nls /etc | wc -l\n```",
            "tool_context": "",
            "cycles": 1
        }

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(create_router(FakeAppGraph()))
    return TestClient(app)

def test_chat_endpoint_response_and_cache(client):
    session_cache.clear()

    payload = {
        "messages": [{"role": "user", "content": "How many files are in /etc?"}],
        "thread_id": "test-thread"
    }

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "```bash\nls /etc | wc -l\n```"

    # On vérifie aussi le cache mis à jour
    assert "test-thread" in session_cache
    cached = session_cache["test-thread"]
    assert cached["current_problem"] == "Count files in /etc."
    assert cached["last_action"] == "```bash\nls /etc | wc -l\n```"

def test_cache_persists_between_calls(client):
    session_cache.clear()

    payload1 = {
        "messages": [{"role": "user", "content": "How many files are in /etc?"}],
        "thread_id": "session-1"
    }
    client.post("/api/chat", json=payload1)

    payload2 = {
        "messages": [{"role": "user", "content": "What was the last command?"}],
        "thread_id": "session-1"
    }
    client.post("/api/chat", json=payload2)

    assert session_cache["session-1"]["cycles"] == 1
    assert session_cache["session-1"]["current_problem"] == "Count files in /etc."

def test_chat_uses_default_thread_id(client):
    session_cache.clear()

    payload = {
        "messages": [{"role": "user", "content": "Any running processes?"}]
    }

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    assert "default" in session_cache
    assert session_cache["default"]["last_action"].startswith("```bash")

def test_chat_with_empty_message_list(client):
    session_cache.clear()

    payload = {
        "messages": [],
        "thread_id": "empty-thread"
    }

    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)

class AppGraphWithCycleIncrement:
    def invoke(self, state, config=None):
        return {
            **state,
            "last_action": "```bash\necho OK\n```",
            "cycles": state.get("cycles", 0) + 1
        }

@pytest.fixture
def client_with_cycle_graph():
    app = FastAPI()
    app.include_router(create_router(AppGraphWithCycleIncrement()))
    return TestClient(app)

def test_cycle_is_incremented(client_with_cycle_graph):
    session_cache.clear()
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thread_id": "cycle-thread"
    }

    client_with_cycle_graph.post("/api/chat", json=payload)
    assert session_cache["cycle-thread"]["cycles"] == 1

    client_with_cycle_graph.post("/api/chat", json=payload)
    assert session_cache["cycle-thread"]["cycles"] == 2

class PartialAppGraph:
    def invoke(self, state, config=None):
        return {
            "last_action": "```bash\necho partial\n```"
        }

@pytest.fixture
def client_with_partial_graph():
    app = FastAPI()
    app.include_router(create_router(PartialAppGraph()))
    return TestClient(app)

def test_partial_result_falls_back_to_context(client_with_partial_graph):
    session_cache["partial"] = {
        "expected_format": "bash",
        "analysis_summary": "Initial summary",
        "tool_history": [],
        "last_output": "",
        "draft_solution": "",
        "current_problem": "Initial task",
        "last_action": "```bash\ninitial\n```",
        "tool_context": "",
        "cycles": 1
    }

    payload = {
        "messages": [{"role": "user", "content": "Just testing"}],
        "thread_id": "partial"
    }

    client_with_partial_graph.post("/api/chat", json=payload)

    assert session_cache["partial"]["analysis_summary"] == "Initial summary"
    assert session_cache["partial"]["last_action"] == "```bash\necho partial\n```"
