from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

router = APIRouter()

class ChatInput(BaseModel):
    prompt: str
    thread_id: str | None = None

# ✅ On définit un placeholder pour l'agent
app_graph = None

@router.post("/api/chat")
async def chat_endpoint(input: ChatInput):
    if app_graph is None:
        return {"error": "Agent not initialized"}
    thread_id = input.thread_id or "default"
    result = app_graph.invoke(
        {"messages": [HumanMessage(content=input.prompt)]},
        config={"thread_id": thread_id}
    )
    return {
        "choices": [
            {"message": {"role": "assistant", "content": result["messages"][-1].content}}
        ]
    }

