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
from reasoning import reasoning_draft_node
from analyse import analyse_problem_node

MAX_CYCLES = 2

llm = model_llm
class AgentState(TypedDict):
    messages: List[BaseMessage]
    expected_format: str
    analysis_summary: str
    current_problem: str
    last_action: str
    draft_solution: str
    tool_context: str
    cycles: int

from termcolor import colored
from langchain_core.messages import SystemMessage, HumanMessage

EXPECTED_FORMATS = """
Think: Explain your reasoning.
Act: bash

```bash
# put your bash code here

Think: Explain why this is an answer.
Act: answer(your_answer_here)

Think: Explain why the task is done.
Act: finish
"""


import re
from langchain_core.messages import SystemMessage

def planner_node(state: AgentState) -> AgentState:
    """
    Décide la prochaine action (ReAct + Toolformer):
    - Continuer à réfléchir
    - Utiliser un outil
    - Passer à la réponse finale
    """
    current_problem = state.get("current_problem", "")
    analysis_summary = state.get("analysis_summary", "")
    draft_solution = state.get("draft_solution", "")
    # user_message = state["messages"][-1].content if state.get("messages") else ""
    previous_tools = state.get("tool_history", [])

    print("\n" + "=" * 60)
    print("[PLANNER] Decision point")
    # print(f"User message: {user_message}")
    # print(f"Previous tools: {previous_tools}")
    print("=" * 60)

    # ✅ Règle 1 : Si le message contient une sortie numérique → aller à reasoning_final
    if re.search(r"\b\d+\b", analysis_summary) and "output of the os" in analysis_summary.lower():
        print("[PLANNER] Detected numeric OS output → switch to final answer mode")
        return {
            **state,
            "plan": {"action": "reasoning_final", "input": "Finalize the answer using expected format"}
        }
    elif re.search(r"Act:\s*answer\([^)]+\)", draft_solution):
        print("[PLANNER] Detected final answer format → switch to reasoning_final")
        return {
            **state,
            "plan": {"action": "reasoning_final", "input": "answer(...) detected"}
        }

    # ✅ Préparer le prompt pour la décision
    prompt = f"""
You are the Orchestrator in a reasoning system.
You have a reasoning draft {draft_solution}
Does this draft help to solve the task {current_problem} or is the answer ?
Decide:
- If it fully answers the question, output reasoning_final
- If it needs improvement, output reasoning_draft
- If it needs command details, ouput linux_doc or search_in_doc
DO NOT OUTPUT ANYTHING ELSE.
"""

    response = llm.invoke([SystemMessage(content=prompt)])
    decision_raw = response.content.strip()

    print("[PLANNER RAW DECISION]")
    print(decision_raw)
    print("-" * 50)

    # ✅ Déterminer l'action
    if "linux_doc" in decision_raw:
        action = "linux_doc"
    elif "search_in_doc" in decision_raw:
        action = "search_in_doc"
    elif "reasoning_final" in decision_raw or "finalize" in decision_raw:
        action = "reasoning_final"
    else:
        action = "reasoning_draft"

    return {
        **state,
        "plan": {"action": action, "input": decision_raw}
    }

from langchain_core.messages import SystemMessage

def reasoning_final_node(state: AgentState) -> AgentState:
    """
    Génère la réponse finale au format strict défini par expected_format.
    Utilise le raisonnement et le contexte accumulé.
    """
    analysis_summary = state.get("analysis_summary", "No summary.")
    draft_solution = state.get("draft_solution", "")
    expected_format = EXPECTED_FORMATS

    tool_context = "\n".join(
        [m.content for m in state["messages"] if "[linux_doc RESULT]" in m.content or "[search_in_doc RESULT]" in m.content]
    )

    print(colored("[DEBUG] Generating FINAL output...", "magenta"))
    print(colored(f"Analysis Summary: {analysis_summary}", "magenta"))
    print(colored(f"Draft Solution: {draft_solution}", "magenta"))
    # ✅ Préparer le prompt strict
#     prompt = f"""
# You are finalizing the solution for the task.

# Context:
# - Task Summary: {analysis_summary}
# - Previous draft: {draft_solution}
# - Tool context: {tool_context if tool_context else "None"}
# - Possible format: {expected_format}

# Rules:
# 1. YOU MUST OUTPUT EXACTLY ONE ACTION.
# 2. Choose only ONE of the above formats. If you use more than one, the answer is INVALID.
# 3. Always start with "Think:" then one "Act:" line according to the chosen format.
# 4. If the task is complete but no numeric answer, use finish.
# 5. Do NOT include multiple Act sections. Do NOT add text outside the format.

# Output ONLY the reasoning and ONE final action.
# """
    prompt = f"""
You are finalizing the solution for the task.

Context:
- Task Summary: {analysis_summary}
- Previous draft: {draft_solution}
- Tool context: {tool_context if tool_context else "None"}
- Expected format: {expected_format}

Rules:
1. YOU MUST OUTPUT EXACTLY ONE ACTION.
2. Choose only ONE of the allowed formats: answer(...), finish, or bash.
3. Always start with "Think:" then one "Act:" line.
4. ONLY use `answer(...)` if you are explicitly provided with a result (e.g., the OS output).
5. NEVER guess or invent the result of a bash command.
6. If the task is complete but no numeric or text output is available, use finish.
7. Do NOT include multiple Act sections. Do NOT add text outside the format.

Output ONLY the reasoning and ONE final action.
"""

    # ✅ Appel LLM
    response = llm.invoke([SystemMessage(content=prompt)])
    final_output = response.content.strip()

    print(colored(f"[FINAL REASONING]\n{final_output}\n{'-'*50}", "magenta"))

    return {
        **state,
        "final_output": final_output,
        "last_action": final_output  # Mémoriser la dernière action pour le prochain cycle
    }

from langgraph.graph import StateGraph, END

# ✅ Création du graphe
graph = StateGraph(AgentState)

# ✅ Ajout des nœuds
graph.add_node("analyze", analyse_problem_node)
graph.add_node("reasoning_draft", reasoning_draft_node)
graph.add_node("planner", planner_node)
graph.add_node("linux_doc", linux_doc_node)
graph.add_node("search_in_doc", search_in_doc_node)
graph.add_node("reasoning_final", reasoning_final_node)

# ✅ Définir les transitions
# graph.add_edge("extract_format", "analyze")
graph.add_edge("analyze", "reasoning_draft")
graph.add_edge("reasoning_draft", "planner")

# ✅ Planner → Décision intelligente
graph.add_conditional_edges("planner", lambda state: state["plan"]["action"], {
    "linux_doc": "linux_doc",
    "search_in_doc": "search_in_doc",
    "reasoning_draft": "reasoning_draft",  # Boucle pour raffiner
    "reasoning_final": "reasoning_final"
})

# ✅ Après un outil → retour au planner
graph.add_edge("linux_doc", "planner")
graph.add_edge("search_in_doc", "planner")

# ✅ Sortie finale
graph.add_edge("reasoning_final", END)

# ✅ Point d'entrée
graph.set_entry_point("analyze")


from routes import create_router
# Compile avec MemorySaver
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

# Crée le router avec le graph
router = create_router(app_graph)

app = FastAPI(title="Linux Agent API")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=11435, reload=True)
