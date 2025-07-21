from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage

router = APIRouter()

class ChatInput(BaseModel):
    prompt: str
    thread_id: str | None = None

# ✅ On définit un placeholder pour l'agent
app_graph = None



# ✅ Modèle aligné sur la structure envoyée par le benchmark
class Message(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    messages: List[Message]
    thread_id: str | None = None

@router.post("/api/chat")
async def chat_endpoint(input: ChatInput):
    # ✅ On récupère le premier message utilisateur
    user_message = input.messages[0].content if input.messages else ""
    thread_id = input.thread_id or "default"

    # ✅ Initialisation de l'état avec tool_history vide
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "tool_history": []  # ✅ Important pour éviter les boucles infinies
    }

    # ✅ On appelle le graph avec la mémoire thread_id
    from main import app_graph  # ⚠ à remplacer par injection si possible
    result = app_graph.invoke(
        initial_state,
        config={"thread_id": thread_id}
    )

    return {
        "choices": [
            {"message": {"role": "assistant", "content": result["messages"][-1].content}}
        ]
    }

    
# @router.post("/api/chat")
# async def chat_endpoint(input: ChatInput):
#     user_message = input.messages[0]["content"]
#     thread_id = input.thread_id or "default"
#     result = app_graph.invoke(
#         {"messages": [HumanMessage(content=user_message)]},
#         config={"thread_id": thread_id}
#     )
#     return {
#         "choices": [
#             {"message": {"role": "assistant", "content": result["messages"][-1].content}}
#         ]
#     }
# async def chat_endpoint(input: ChatInput):
#     if app_graph is None:
#         return {"error": "Agent not initialized"}
#     thread_id = input.thread_id or "default"
#     result = app_graph.invoke(
#         {"messages": [HumanMessage(content=input.prompt)]},
#         config={"thread_id": thread_id}
#     )
#     return {
#         "choices": [
#             {"message": {"role": "assistant", "content": result["messages"][-1].content}}
#         ]
#     }

