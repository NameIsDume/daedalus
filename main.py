from fastapi import FastAPI
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
import uvicorn
from model import model_llm
from tools import linux_doc_node, search_in_doc_node
from routes import router

# ==========================================
# 1. Définir les Schemas Pydantic
# ==========================================
class PlanOutput(BaseModel):
    action: Literal["linux_doc", "search_in_doc", "finish"]
    input: Dict[str, str]

class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: PlanOutput
    tool_history: List[str]

class ChatInput(BaseModel):
    prompt: str
    thread_id: str = "default"

# ==========================================
# 3. Initialiser le modèle
# ==========================================
llm = model_llm  # Remplace par ton modèle local (ex: Ollama)


# def extract_json(text: str) -> str:
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if match:
#         return match.group(0)
#     raise ValueError("No JSON found in LLM output")

import re
import json

def extract_json(text: str) -> str:
    # 1. Supprimer les backticks et "json"
    clean_text = re.sub(r"```(json)?", "", text).strip()

    # 2. Rechercher le premier bloc JSON
    match = re.search(r"\{[\s\S]*\}", clean_text)
    if match:
        return match.group(0)

    # 3. Si pas trouvé, lever une exception pour fallback
    raise ValueError(f"No valid JSON found in LLM output: {text[:200]}...")

# def extract_json(text: str) -> str:
#     """
#     Extrait le dernier JSON valide dans une chaîne contenant du raisonnement
#     et potentiellement plusieurs blocs JSON. Si aucun JSON valide, renvoie un JSON fallback.
#     """
#     # Récupère toutes les occurrences entre accolades
#     candidates = re.findall(r"\{.*?\}", text, re.DOTALL)

#     # Vérifie chaque occurrence en partant de la fin
#     for candidate in reversed(candidates):
#         try:
#             json.loads(candidate)  # Teste la validité JSON
#             return candidate
#         except json.JSONDecodeError:
#             continue

#     # Fallback si aucun JSON valide trouvé
#     print("[WARN] Aucun JSON valide trouvé, utilisation du fallback.")
#     return json.dumps({
#         "action": "finish",
#         "input": {"answer": "Could not parse LLM response."}
#     })

# ==========================================
# Analyze node
# ==========================================
from langchain_core.messages import SystemMessage, HumanMessage
from termcolor import colored

def formatter_node(state: AgentState) -> AgentState:
    """
    Formate la réponse finale en respectant le format du benchmark :
    - Si la réponse contient une commande → bash
    - Sinon → answer
    """
    analysis_summary = state.get("analysis", "No analysis available.")
    final_input = state["plan"].input
    answer = final_input.get("answer", "").strip()

    # Heuristique : si "ls", "find", "cat", etc. dans la réponse → bash
    if any(cmd in answer for cmd in ["ls", "find", "cat", "grep", "wc", "chmod"]):
        formatted = f"""Think: {analysis_summary}

Act: bash

```bash
{answer}
```"""
    else:
        formatted = f"""Think: {analysis_summary}

Act: answer({answer})"""

    print(colored(f"[FINAL OUTPUT]\n{formatted}\n{'-'*50}", "cyan", attrs=["bold"]))
    return {"messages": state["messages"] + [HumanMessage(content=formatted)]}

from termcolor import colored
from langchain_core.messages import HumanMessage
from typing import Dict

def analyze_task(state: Dict) -> Dict:
    """
    Analyse la consigne utilisateur et produit un résumé simple.
    """
    user_message = state["messages"][-1].content  # Dernier message utilisateur

    # ✅ Prompt pour LLM
    prompt = f"""
Summarize the user's request in one short sentence. Do NOT provide solution, just describe the task clearly.

User message: "{user_message}"
Output format:
Summary: <one concise sentence>
"""

    # ✅ Appel LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    summary_text = response.content.strip()

    # ✅ Debug
    print(colored(f"[DEBUG] Analysis Summary: {summary_text}", "blue", attrs=["bold"]))

    # ✅ Ajout dans l'état pour usage des autres nodes
    return {
        "analysis_summary": summary_text,
        "messages": state["messages"] + [HumanMessage(content=f"[ANALYSIS]\n{summary_text}")]
    }

#############
from pydantic import BaseModel
from typing import Dict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from termcolor import colored
import re

# ✅ Modèle Pydantic pour valider la sortie
class PlanOutput(BaseModel):
    action: Literal["linux_doc", "search_in_doc", "finish"]
    input: Dict[str, str]

def extract_json(text: str) -> str:
    """Extrait le premier objet JSON valide trouvé dans un texte."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    raise ValueError("No JSON found")

def planner_node(state: Dict) -> Dict:
    """
    Étape 2 : Planner.
    À partir de l'`analysis_summary`, décide quelle action effectuer et construit un JSON.
    """
    analysis_summary = state.get("analysis_summary", "")
    previous_tools = state.get("tool_history", [])

    prompt = f"""
You are a Linux command assistant that plans the next step.

Task Summary: "{analysis_summary}"

Available actions:
- linux_doc(command): Fetch the manual page of a Linux command.
- search_in_doc(command, keyword): Search a keyword inside the manual page.
- finish: Provide the final answer.

Rules:
1. If this is the first step, choose linux_doc with the most relevant command.
2. Use search_in_doc if you already fetched a manual and need a specific flag/keyword.
3. Use finish if you have enough information to answer.
4. Do NOT repeat the same tool twice.
5. Output only JSON, no text outside JSON.

Examples:
{{"action": "linux_doc", "input": {{"command": "ls"}}}}
{{"action": "search_in_doc", "input": {{"command": "ls", "keyword": "hidden"}}}}
{{"action": "finish", "input": {{"answer": "Use ls -a to display hidden files."}}}}

Now decide:
"""

    # ✅ Génération via LLM
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Previous tools used: {previous_tools}")
    ])

    raw_output = response.content.strip()
    print(colored(f"[DEBUG] Raw Planner Response:\n{raw_output}", "yellow", attrs=["bold"]))

    # ✅ Extraction JSON avec fallback
    try:
        json_text = extract_json(raw_output)
        parsed = PlanOutput.model_validate_json(json_text)
    except Exception as e:
        print(colored(f"[WARN] Invalid JSON → fallback mode: {e}", "red"))
        parsed = PlanOutput(action="finish", input={"answer": "Could not parse LLM response."})

    print(colored(f"[PLANNER DECISION] {parsed.action} → input: {parsed.input}", "green", attrs=["bold"]))

    return {
        "plan": parsed,
        "messages": state["messages"] + [HumanMessage(content=f"[PLANNER OUTPUT]\n{parsed.model_dump_json()}")]
    }


# ==========================================
# 4. LLM Planner Node
# ==========================================
def llm_planner(state: Dict) -> Dict:
    """Planifie le prochain outil ou termine."""
    messages = state["messages"]
    prompt = """
You are an autonomous Linux assistant. You have two tools:
- linux_doc(command): Fetch the manual page of a Linux command.
- search_in_doc(command, keyword): Search for a specific keyword inside the manual page of a command.

Important Rules:
- First, check if the user question is related to Linux commands or terminal usage.
- If the question is NOT about Linux or shell commands, answer immediately and do NOT call any tool.
- If the question IS about Linux commands:
    1. You MUST call linux_doc first for the most relevant command.
    2. After calling linux_doc, if needed, use search_in_doc ONLY if the question asks about a specific option or flag, or if you need to locate a specific keyword in the manual.
    3. NEVER invent keywords that do not appear in the user question.
    4. If the answer is clear after linux_doc, skip search_in_doc and go to finish.
- NEVER return finish as your first action in a Linux-related question.

Output ONLY JSON, with no additional text:
{"action": "linux_doc"|"search_in_doc"|"finish", "input": {...}}
After reasoning, OUTPUT ONLY a single valid JSON object on a new line.
Do NOT include anything else after the JSON.

Examples:
{"action": "linux_doc", "input": {"command": "ls"}}
{"action": "search_in_doc", "input": {"command": "ls", "keyword": "hidden"}}
{"action": "finish", "input": {"answer": "Use ls -a to display hidden files."}}
"""
    response = llm.invoke([SystemMessage(content=prompt)] + messages)

    # ✅ Extraire uniquement le JSON
    json_text = extract_json(response.content)
    parsed = PlanOutput.model_validate_json(json_text)
    print(colored("\n[DEBUG] Raw LLM Response:\n", "yellow", attrs=["bold"]))
    print(response.content)
    print(colored("\n[DEBUG] End of Raw Response\n", "yellow", attrs=["bold"]))
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
graph = StateGraph(AgentState)

def route_from_plan(state: AgentState) -> str:
    """
    Détermine le prochain nœud en fonction du plan.
    Ajoute une logique anti-boucle basée sur l'historique des outils.
    """
    action = state["plan"].action
    history = state.get("tool_history", [])

    # Si l'action prévue est déjà exécutée → on passe directement à formatter
    if action in history:
        print(colored(f"[WARN] Action '{action}' déjà exécutée. Passage à formatter.", "red", attrs=["bold"]))
        return "formatter"

    # Logique normale
    if action == "linux_doc":
        return "linux_doc"
    elif action == "search_in_doc":
        return "search_in_doc"
    elif action == "finish":
        return "formatter"

    # Fallback (par sécurité)
    return "formatter"

graph.add_node("analyzer", analyze_task)
graph.add_node("planner", planner_node)
graph.add_node("linux_doc", linux_doc_node)
graph.add_node("search_in_doc", search_in_doc_node)
graph.add_node("formatter", formatter_node)

# Edges
graph.add_edge("analyzer", "planner")  # L'analyse mène au planner
graph.add_conditional_edges("planner", route_from_plan, {
    "linux_doc": "linux_doc",
    "search_in_doc": "search_in_doc",
    "formatter": "formatter"
})
graph.add_edge("linux_doc", "planner")
graph.add_edge("search_in_doc", "planner")
graph.add_edge("formatter", END)

# Entry point
graph.set_entry_point("analyzer")

# Compile avec MemorySaver
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

import routes
routes.app_graph = app_graph

app = FastAPI(title="Linux Agent API")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=11435, reload=True)