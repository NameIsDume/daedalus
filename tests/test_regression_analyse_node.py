def test_regression_summary_for_simple_problem():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

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

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    assert "how many" in summary or "number of" in summary, \
        f"Unexpected summary: {summary}"

def test_regression_summary_does_not_include_solution():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage
    import re

    state = {
        "messages": [HumanMessage(content="List all python packages installed.")],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    assert not re.search(r"\b\d+\b", summary), "Summary includes numeric result"
    assert "pip" not in summary and "bash" not in summary and "`" not in summary, \
        "Summary seems to include a command"

def test_regression_output_numeric_is_recognized():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="The output of the OS:\n42")],
        "expected_format": "",
        "analysis_summary": "The user wants to count the files in /etc.",
        "current_problem": "Count the number of files in /etc.",
        "last_action": "```bash\nls /etc | wc -l\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()
    expected_keywords = ["indicates", "42", "files"]
    matches = all(keyword in summary for keyword in expected_keywords)

    assert matches, f"Unexpected interpretation: {summary}"

def test_regression_error_output_is_flagged():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="The output of the OS:\nbash: command not found")],
        "expected_format": "",
        "analysis_summary": "The user tried to run a command.",
        "current_problem": "Execute a specific command.",
        "last_action": "```bash\nfoobar\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    assert any(keyword in summary for keyword in ["error", "failed", "not found", "invalid"]), \
        f"Expected an error interpretation, got: {summary}"

def test_regression_summary_initial_has_no_solution():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="I want to count the number of files in /tmp")],
        "expected_format": "",
        "analysis_summary": "",
        "current_problem": "",
        "last_action": "",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 0,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    forbidden_keywords = ["ls", "wc", "grep", "find", "bash"]
    assert not any(cmd in summary for cmd in forbidden_keywords), \
        f"Expected no solution leaked into summary, got: {summary}"

def test_regression_output_float_number_is_recognized():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="The output of the OS:\n42.0")],
        "expected_format": "",
        "analysis_summary": "The user wants to count the files in /opt.",
        "current_problem": "Count the number of files in /opt.",
        "last_action": "```bash\nls /opt | wc -l\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    expected_keywords = ["indicates", "42", "files"]
    assert all(k in summary for k in expected_keywords), f"Unexpected interpretation: {summary}"

def test_regression_irrelevant_output_is_detected():
    from analyse import analyse_problem_node
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="The output of the OS:\nWelcome to Ubuntu 22.04 LTS")],
        "expected_format": "",
        "analysis_summary": "The user wants to list all installed packages.",
        "current_problem": "List all installed packages.",
        "last_action": "```bash\napt list --installed\n```",
        "draft_solution": "",
        "tool_context": "",
        "cycles": 1,
    }

    result = analyse_problem_node(state)
    summary = result["analysis_summary"].lower()

    keywords_indicating_irrelevance = ["unrelated", "not relevant", "not related", "irrelevant", "does not help"]

    assert any(k in summary for k in keywords_indicating_irrelevance), f"Expected irrelevance detection. Got: {summary}"
