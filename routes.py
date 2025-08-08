from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage

session_cache: Dict[str, Dict] = {}

class Message(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    messages: List[Message]
    thread_id: str | None = None

def create_router(app_graph):
    router = APIRouter()

    @router.post("/api/chat")
    async def chat_endpoint(input: ChatInput):
        user_message = input.messages[-1].content if input.messages else ""
        thread_id = input.thread_id or "default"

        # print(f"Received chat input: {user_message} (thread_id: {thread_id})")
        # print("#############################################")

        context = session_cache.get(thread_id, {
            "expected_format": "",
            "analysis_summary": "",
            "tool_history": [],
            "last_output": ""
        })

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "expected_format": context.get("expected_format", ""),
            "analysis_summary": context.get("analysis_summary", ""),
            "tool_history": context.get("tool_history", []),
            "draft_solution": context.get("draft_solution", ""),
            "current_problem": context.get("current_problem", ""),
            "last_action": context.get("last_action", ""),
            "tool_context": context.get("tool_context", ""),
            "cycles": context.get("cycles", 0)
        }

        result = app_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        session_cache[thread_id] = {
            "expected_format": result.get("expected_format", context["expected_format"]),
            "analysis_summary": result.get("analysis_summary", context["analysis_summary"]),
            "tool_history": result.get("tool_history", context["tool_history"]),
            "draft_solution": result.get("draft_solution", context.get("draft_solution", None)),
            "current_problem": result.get("current_problem", context.get("current_problem", None)),
            "last_action": result.get("last_action", context.get("last_action", None)),
            "tool_context": result.get("tool_context", context.get("tool_context", None)),
            "cycles": result.get("cycles", context.get("cycles", 0))
        }

        return {
            "choices": [
                {"message": {"role": "assistant", "content": result["last_action"]}}
            ]
        }

    return router
