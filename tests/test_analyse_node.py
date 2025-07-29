import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from analyse import AgentState
import analyse

def test_analyse_node_first_interaction_basic(monkeypatch):
    from analyse import analyse_node_first_interaction
    from langchain_core.messages import HumanMessage

    def fake_invoke(messages, *args, **kwargs):
        class FakeResponse:
            content = "Count the number of files in the /etc directory."
        return FakeResponse()

    import analyse
    monkeypatch.setattr(analyse.llm.__class__, "invoke", fake_invoke)

    # Crée un état initial
    state: AgentState = {
        "messages": [HumanMessage(content="How many files are in the /etc directory?")],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0
    }

    new_state = analyse_node_first_interaction(state)

    assert new_state["analysis_summary"] == "Count the number of files in the /etc directory."
    assert new_state["current_problem"] == new_state["analysis_summary"]
    assert len(new_state["messages"]) == 2
    assert isinstance(new_state["messages"][-1], HumanMessage)
    assert new_state["messages"][-1].content == "How many files are in the /etc directory?"

def test_analyse_node_previous_summary(monkeypatch):
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    def fake_invoke(messages, *args, **kwargs):
        class FakeResponse:
            content = "The output indicates 42 files were found."
        return FakeResponse()

    monkeypatch.setattr(analyse.llm.__class__, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content="The output of the OS: 42")],
        "expected_format": "",
        "analysis_summary": "User wants to count files in /home",
        "current_problem": "User wants to count files in /home",
        "last_action": "ls /home | wc -l",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    new_state = analyse_problem_node(state)

    assert isinstance(new_state["analysis_summary"], str)
    assert "42" in new_state["analysis_summary"]
    assert len(new_state["messages"]) == 2

def test_start_new_task_resets_context(monkeypatch):
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    def fake_invoke(messages, *args, **kwargs):
        class FakeResponse:
            content = "The user is starting a new problem on a different operating system."
        return FakeResponse()

    import analyse
    monkeypatch.setattr(analyse.llm.__class__, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content="Now, I will start a new problem in a new OS.")],
        "expected_format": "answer(x)",
        "analysis_summary": "Old summary",
        "current_problem": "Old problem",
        "last_action": "answer(42)",
        "draft_solution": "Old draft",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_problem_node(state)

    assert result["current_problem"] == "The user is starting a new problem on a different operating system."
    assert result["last_action"] == ""
    assert result["draft_solution"] == ""

def test_analyse_node_first_interaction_without_keyword(monkeypatch):
    from analyse import analyse_node_first_interaction
    from langchain_core.messages import HumanMessage

    def fake_invoke(messages, *args, **kwargs):
        class FakeResponse:
            content = "Count how many files are in the /etc directory."
        return FakeResponse()

    import analyse
    monkeypatch.setattr(analyse.llm.__class__, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content="How many files are in the /etc directory?")],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = analyse_node_first_interaction(state)
    assert result["analysis_summary"] == "Count how many files are in the /etc directory."
    assert result["current_problem"] == result["analysis_summary"]

def test_analyse_node_previous_summary_error(monkeypatch):
    from analyse import analyse_node_previous_summary
    from langchain_core.messages import HumanMessage

    def fake_invoke(messages, *args, **kwargs):
        class FakeResponse:
            content = "The output is an error message, indicating a failure in command execution."
        return FakeResponse()

    import analyse
    monkeypatch.setattr(analyse.llm.__class__, "invoke", fake_invoke)

    state = {
        "messages": [HumanMessage(content="bash: command not found")],
        "expected_format": "bash",
        "analysis_summary": "Trying to find number of processes",
        "current_problem": "Find number of processes",
        "last_action": "ps aux | wc -l",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_node_previous_summary(state)
    assert "error" in result["analysis_summary"].lower()

def test_start_new_task_does_nothing():
    from analyse import start_new_task_if_needed
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="Show me the disk usage")],
        "expected_format": "answer(x)",
        "analysis_summary": "Old summary",
        "current_problem": "Old problem",
        "last_action": "answer(42)",
        "draft_solution": "Old draft",
        "tool_context": "",
        "cycles": 1,
    }

    result = start_new_task_if_needed(state)
    assert result == state
