from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage

# ----------------------
# ✅ Cache en mémoire
# ----------------------
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

        # ----------------------
        # ✅ Récupération du contexte existant
        # ----------------------
        context = session_cache.get(thread_id, {
            "expected_format": "",
            "analysis_summary": "",
            "tool_history": [],
            "last_output": ""
        })

        # ----------------------
        # ✅ Préparation de l'état initial pour app_graph
        # ----------------------
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "tool_history": context["tool_history"],
            "draft_solution": "",
            "cycles": 0,
            "analysis_summary": context["analysis_summary"],
            "expected_format": context["expected_format"],
            "last_output": context["last_output"]
        }

        # ✅ Exécution du graphe
        result = app_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # ----------------------
        # ✅ Mise à jour du cache
        # ----------------------
        session_cache[thread_id] = {
            "expected_format": result.get("expected_format", context["expected_format"]),
            "analysis_summary": result.get("analysis_summary", context["analysis_summary"]),
            "tool_history": result.get("tool_history", context["tool_history"]),
            "last_output": user_message  # Dernière sortie reçue du "système"
        }

        # ✅ Retour au client
        return {
            "choices": [
                {"message": {"role": "assistant", "content": result["messages"][-1].content}}
            ]
        }

    return router
