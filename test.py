from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing import Literal, Dict
import httpx
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from termcolor import colored
from model import model_llm  # Assurez-vous que ce modèle est correctement importé
            # Le dernier plan du LLM (facultatif pour démarrage)

# ==========================================
# 1. Définir les Schemas Pydantic
# ==========================================
class PlanOutput(BaseModel):
    action: Literal["linux_doc", "search_in_doc", "finish"]
    input: Dict[str, str]

class AgentState(TypedDict):
    messages: List[BaseMessage]   # Historique des messages
    plan: PlanOutput  
# ==========================================
# 2. Tools
# ==========================================
@tool
def linux_doc(command: str) -> str:
    """Fetch Linux manual page for a command."""
    base_command = command.strip().split()[0]
    url = f"http://localhost:9000/get_doc?command={base_command}"
    resp = httpx.get(url, timeout=10)
    data = resp.json()
    if "error" in data:
        return f"No documentation found for '{base_command}'"
    prefix = colored("[TOOL]", "magenta")
    print(f"{prefix} has been called")
    return data['full_doc'][:1500]

@tool
def search_in_doc(command: str, keyword: str) -> str:
    """Search for a keyword inside the Linux manual of a command."""
    base_command = command.strip().split()[0]
    url = f"http://localhost:9000/get_doc?command={base_command}"
    resp = httpx.get(url, timeout=10)
    data = resp.json()
    if "error" in data:
        return f"No documentation found for '{base_command}'"
    matches = [line for line in data['full_doc'].splitlines() if keyword.lower() in line.lower()]
    if not matches:
        return f"No matches found for '{keyword}'"
    prefix = colored("[TOOL]", "magenta")
    print(f"{prefix} has been called")
    return "\n".join(matches[:10])

def linux_doc_node(state: AgentState) -> AgentState:
    command = state["plan"].input["command"]
    result = linux_doc.run(command)  # On appelle la fonction tool
    return {"messages": state["messages"] + [HumanMessage(content=f"[TOOL linux_doc]\n{result}")]}

def search_in_doc_node(state: AgentState) -> AgentState:
    command = state["plan"].input["command"]
    keyword = state["plan"].input["keyword"]
    result = search_in_doc.run(command, keyword)
    return {"messages": state["messages"] + [HumanMessage(content=f"[TOOL search_in_doc]\n{result}")]}


# ==========================================
# 3. Initialiser le modèle
# ==========================================
llm = model_llm  # Remplace par ton modèle local (ex: Ollama)

import re

def extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No JSON found in LLM output")

# ==========================================
# 4. LLM Planner Node
# ==========================================
def llm_planner(state: Dict) -> Dict:
    """Planifie le prochain outil ou termine."""
    messages = state["messages"]
    prompt = """
You are an autonomous Linux assistant. Tools:
- linux_doc(command)
- search_in_doc(command, keyword)

Important Rules:
1. You MUST call linux_doc first for the command you think is relevant, even if you know the answer.
2. After calling linux_doc, if necessary, use search_in_doc to find specific options or keywords.
3. Only after using linux_doc (and search_in_doc if needed), you can finish.
4. NEVER return finish as your first action.

Output ONLY JSON, no extra text:
{"action": "linux_doc"|"search_in_doc"|"finish", "input": {...}}

Examples:
{"action": "linux_doc", "input": {"command": "ls"}}
{"action": "search_in_doc", "input": {"command": "ls", "keyword": "hidden"}}
{"action": "finish", "input": {"answer": "Use ls -a to display hidden files."}}
"""
    response = llm.invoke([SystemMessage(content=prompt)] + messages)

    # ✅ Extraire uniquement le JSON
    json_text = extract_json(response.content)
    parsed = PlanOutput.model_validate_json(json_text)

    print(colored(f"[PLANNER DECISION] {parsed.action} → input: {parsed.input}", "green", attrs=["bold"]))
    return {"plan": parsed}

# ==========================================
# 5. Node Final Answer
# ==========================================
def final_answer(state: Dict) -> Dict:
    """Construit la réponse finale."""
    answer = state["plan"].input.get("answer", "No final answer provided.")
    return {"messages": [HumanMessage(content=f"Final Answer: {answer}")]}

# ==========================================
# 6. Construire le Graph LangGraph
# ==========================================


# ✅ Définir le schema du state

# ✅ Créer le graph avec le schema
graph = StateGraph(AgentState)

graph.add_node("planner", llm_planner)
graph.add_node("linux_doc", linux_doc_node)
graph.add_node("search_in_doc", search_in_doc_node)
graph.add_node("final_answer", final_answer)

# Edges
def route_from_plan(state: AgentState) -> str:
    """Détermine le prochain nœud en fonction du plan."""
    if state["plan"].action == "linux_doc":
        return "linux_doc"
    elif state["plan"].action == "search_in_doc":
        return "search_in_doc"
    elif state["plan"].action == "finish":
        return "final_answer"
    return "final_answer"  # fallback

# 2. Ajouter des conditional edges
graph.add_conditional_edges(
    "planner",
    route_from_plan,
    {
        "linux_doc": "linux_doc",
        "search_in_doc": "search_in_doc",
        "final_answer": "final_answer"
    }
)

graph.add_edge("linux_doc", "planner")
graph.add_edge("search_in_doc", "planner")
graph.add_edge("final_answer", END)
graph.set_entry_point("planner")

# Compile
checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)
# app_graph = graph.compile()

# ==========================================
# 7. FastAPI Server
# ==========================================
app = FastAPI(title="Linux Agent API")

class ChatInput(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat_endpoint(input: ChatInput):
    result = app_graph.invoke({"messages": [HumanMessage(content=input.prompt)]})
    return {"choices": [{"message": {"role": "assistant", "content": result["messages"][-1].content}}]}

# ==========================================
# 8. Lancement avec uvicorn
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11435, reload=False)
