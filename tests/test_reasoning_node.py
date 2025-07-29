import reasoning

class FakeLLM:
    def invoke(self, messages):
        class FakeResponse:
            content = "Think: I need to count the files.\n\nAct: ```bash\nls /etc | wc -l\n```"
        return FakeResponse()

def test_reasoning_draft_first_interaction(monkeypatch):
    monkeypatch.setattr(reasoning, "llm", FakeLLM())

    state = {
        "messages": [],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "Count the number of files in /etc.",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = reasoning.reasoning_draft_first_interaction(state)
    assert "Think:" in result["draft_solution"]

import reasoning

class FakeLLM:
    def invoke(self, messages):
        class FakeResponse:
            content = "Think: I need to count the files.\n\nAct: ```bash\nls /etc | wc -l\n```"
        return FakeResponse()

def test_reasoning_draft_first_interaction(monkeypatch):
    monkeypatch.setattr(reasoning, "llm", FakeLLM())

    state = {
        "messages": [],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "Count the number of files in /etc.",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = reasoning.reasoning_draft_first_interaction(state)
    assert "Think:" in result["draft_solution"]
    assert "Act:" in result["draft_solution"]

def test_reasoning_draft_multiple_steps_shortcut(monkeypatch):
    monkeypatch.setattr(reasoning, "llm", FakeLLM())

    state = {
        "messages": [
            reasoning.HumanMessage(content="The output of the OS:\n42")
        ],
        "expected_format": "",
        "analysis_summary": "Output: 42",
        "current_problem": "How many files are in /etc?",
        "last_action": "```bash\nls /etc | wc -l\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = reasoning.reasoning_draft_multiple_steps(state)
    assert "Act: answer(42)" in result["draft_solution"]

def test_reasoning_draft_node_dispatch(monkeypatch):
    monkeypatch.setattr(reasoning, "llm", FakeLLM())

    state = {
        "messages": [],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "List processes.",
        "last_action": None,
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = reasoning.reasoning_draft_node(state)
    assert "Think:" in result["draft_solution"]
    assert "Act:" in result["draft_solution"]
