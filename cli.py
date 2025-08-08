import uuid
from langchain_core.messages import HumanMessage

session_cache = globals().get("session_cache", {})

def run_cli(app_graph, thread_id: str | None = None, output_mode: str = "last_action"):
    print("CLI mode - write 'exit' to leave\n")

    thread_id = thread_id or f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    def get_role(m):

        t = getattr(m, "type", None)
        if t:
            return {"ai": "assistant", "human": "user", "system": "system"}.get(t, t)
        if isinstance(m, dict):
            return m.get("role") or m.get("type")
        return None

    def get_content(m):
        if hasattr(m, "content"):
            return m.content
        if isinstance(m, dict):
            return m.get("content")
        return None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("exit", "quit"):
            break

        context = session_cache.get(thread_id, {
            "expected_format": "",
            "analysis_summary": "",
            "tool_history": [],
            "last_output": "",
            "draft_solution": "",
            "current_problem": "",
            "last_action": "",
            "tool_context": "",
            "cycles": 0,
        })

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "expected_format": context.get("expected_format", ""),
            "analysis_summary": context.get("analysis_summary", ""),
            "tool_history": context.get("tool_history", []),
            "draft_solution": context.get("draft_solution", ""),
            "current_problem": context.get("current_problem", ""),
            "last_action": context.get("last_action", ""),
            "tool_context": context.get("tool_context", ""),
            "cycles": context.get("cycles", 0),
        }

        result = app_graph.invoke(initial_state, config=config)

        session_cache[thread_id] = {
            "expected_format": result.get("expected_format", context["expected_format"]),
            "analysis_summary": result.get("analysis_summary", context["analysis_summary"]),
            "tool_history": result.get("tool_history", context["tool_history"]),
            "draft_solution": result.get("draft_solution", context.get("draft_solution", "")),
            "current_problem": result.get("current_problem", context.get("current_problem", "")),
            "last_action": result.get("last_action", context.get("last_action", "")),
            "tool_context": result.get("tool_context", context.get("tool_context", "")),
            "cycles": result.get("cycles", context.get("cycles", 0)),
        }

        if output_mode == "assistant":
            msgs = result["messages"] if isinstance(result, dict) and "messages" in result else []
            assistant_msg = next((m for m in reversed(msgs) if get_role(m) == "assistant"), None)
            if assistant_msg:
                print("Agent:", get_content(assistant_msg))
            else:
                print("Agent:", result.get("last_action", result))
        else:
            print("Agent:", result.get("last_action", result))
