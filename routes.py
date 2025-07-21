from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage

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

        print(f"Received chat input: {user_message} (thread_id: {thread_id})")
        print("#############################################")

        # ✅ Construire l'état initial avec le message utilisateur
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "tool_history": [],
            "draft_solution": "",
            "cycles": 0,
            "analysis_summary": "",
            "expected_format": ""
        }

        # ✅ Appel du graph
        result = app_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        return {
            "choices": [
                {"message": {"role": "assistant", "content": result["messages"][-1].content}}
            ]
        }


    return router
