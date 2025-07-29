import pytest
from reasoning import reasoning_draft_node, AgentState
from langchain_core.messages import HumanMessage

def test_regression_summary_for_simple_problem():
    from reasoning import reasoning_draft_multiple_steps
    from langchain_core.messages import HumanMessage

    state: AgentState = {
        "messages": [HumanMessage(content="The output of the OS:\n220")],
        "expected_format": "",
        "analysis_summary": "The output indicates that there are 220 files in the /etc directory",
        "current_problem": "How many files are in /etc?",
        "last_action": "```bash\nls /etc | wc -l\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = reasoning_draft_multiple_steps(state)
    draft_solution = result["draft_solution"]

    assert "answer(220)" in draft_solution

def test_regression_reasoning_first_interaction():
    from reasoning import reasoning_draft_first_interaction

    state = {
        "messages": [HumanMessage(content="How can I list all files in /etc?")],
        "expected_format": "",
        "analysis_summary": "The user wants to list files.",
        "current_problem": "List all files in /etc.",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = reasoning_draft_first_interaction(state)
    draft = result["draft_solution"]

    assert draft.startswith("Think:"), f"Missing Think: section → {draft}"
    assert "Act:" in draft, f"Missing Act: section → {draft}"
    assert "```bash" in draft, f"Expected a bash block → {draft}"

def test_regression_reasoning_node_first_call():
    from reasoning import reasoning_draft_node

    state = {
        "messages": [HumanMessage(content="Find the number of files in /opt.")],
        "expected_format": "",
        "analysis_summary": "User wants a count of files in /opt.",
        "current_problem": "Count files in /opt.",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = reasoning_draft_node(state)
    draft = result["draft_solution"]

    assert draft.startswith("Think:")
    assert "```bash" in draft or "Act:" in draft

def test_regression_reasoning_node_continues_and_answers():
    from reasoning import reasoning_draft_node

    state = {
        "messages": [HumanMessage(content="The output of the OS:\n19")],
        "expected_format": "",
        "analysis_summary": "There are 19 users.",
        "current_problem": "How many users are defined?",
        "last_action": "```bash\ncut -d: -f1 /etc/passwd | wc -l\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = reasoning_draft_node(state)
    draft = result["draft_solution"]

    assert "answer(19)" in draft
