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

MAX_CYCLES = 2

llm = model_llm
class AgentState(TypedDict):
    messages: List[BaseMessage]
    expected_format: str
    analysis_summary: str
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

def analyze_problem_node(state: AgentState) -> AgentState:
    """
    Analyse le dernier message utilisateur (ou sortie OS) dans un contexte multi-étapes.
    Prend en compte :
    - Objectif initial (analysis_summary)
    - Dernière action réellement exécutée
    - Sortie reçue
    """

    user_message = state["messages"][-1].content if state.get("messages") else ""
    previous_summary = state.get("analysis_summary", "")
    last_action = state.get("last_action", None)  # ✅ Nouvelle variable clé

    print(colored(f"[DEBUG] Analyzing user message:\n{user_message}\n{'-'*50}", "cyan"))

    analysis_summary = previous_summary or None
    probable_answer = None
    error_detected = False

    # ✅ Cas 1 : Si la sortie est un nombre → probable réponse finale
    if user_message.strip().isdigit():
        probable_answer = user_message.strip()
        print(colored(f"[DEBUG] Detected probable answer: {probable_answer}", "yellow"))

    # ✅ Cas 2 : Détection d'erreurs système
    elif any(err in user_message.lower() for err in ["not found", "no such", "permission denied", "invalid option"]):
        error_detected = True
        print(colored(f"[DEBUG] Detected OS error in response.", "red"))

    # ✅ Génération du résumé contextuel
    if not previous_summary:
        # Premier message → définir le but initial
        prompt = f"""
Summarize the user's problem in one short sentence.
User message:
---
{user_message}
---
Rules:
- Do NOT propose a solution.
- Max 30 words.
"""
        response = llm.invoke([SystemMessage(content=prompt)])
        analysis_summary = response.content.strip()
        print(colored(f"[DEBUG] Initial Analysis Summary: {analysis_summary}", "cyan"))

    else:
        # Contexte multi-étapes → interprétation par rapport à la dernière action
        prompt = f"""
You are analyzing the output of an executed command in a multi-step reasoning process.

Previous goal: "{previous_summary}"
Last executed action: "{last_action if last_action else 'None'}"
System output: "{user_message}"

Explain in ONE short sentence what this output means in relation to the goal.
Examples:
- If it's just a number → It's probably the result (file count).
- If it's an error → Command failed, needs correction.
- If unrelated → Output irrelevant to goal.

Return only the interpretation, no extra text.
"""
        response = llm.invoke([SystemMessage(content=prompt)])
        analysis_summary = response.content.strip()
        print(colored(f"[DEBUG] Contextual Interpretation: {analysis_summary}", "cyan"))

    # ✅ Met à jour l'état
    return {
        **state,
        "analysis_summary": analysis_summary,
        "probable_answer": probable_answer,
        "error_detected": error_detected,
        "messages": state["messages"] + [HumanMessage(content=f"[Analysis Summary]\n{analysis_summary}")]
    }

def reasoning_draft_node(state: AgentState) -> AgentState:
    """
    Génère un raisonnement initial et propose une action candidate.
    Utilise les formats extraits dans expected_format pour respecter le benchmark.
    """
    analysis_summary = state.get("analysis_summary", "No summary available.")
    user_message = state["messages"][-1].content if state.get("messages") else ""

    print(colored("[DEBUG] Generating draft reasoning...", "yellow"))
    print(colored(f"Analysis Summary: {analysis_summary}", "yellow"))
    print(colored(f"Last User Message: {user_message}", "yellow"))

    prompt = f"""
You are an expert Linux assistant following a strict output format.

Your task:
- Solve the user's problem by reasoning step by step.
- Then propose ONE next action (bash command or final answer).

Context:
- Current Task: {analysis_summary}
- Last User Message: {user_message}

Rules:
1. Start with "Think:" and explain your reasoning clearly.
2. Then output the action using the correct format (Act: ...).
3. Do NOT add extra text outside the format.
4. Only ONE action per response.
5. If the user's message looks like an OS command output (e.g., numeric value or short text), interpret it as the result of the last executed command.
6. If that result solves the task, return the answer using the format:
   Act: answer(<value>)
7. If the task is not solved yet, or more steps are needed, return a bash command
"""

    # ✅ Appel LLM
    response = llm.invoke([SystemMessage(content=prompt)])
    draft_solution = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_solution}\n{'-'*50}", "yellow"))

    return {
        **state,
        "draft_solution": draft_solution,
        "messages": state["messages"] + [HumanMessage(content=f"[Draft Solution]\n{draft_solution}")]
    }

import re
from langchain_core.messages import SystemMessage

def planner_node(state: AgentState) -> AgentState:
    """
    Décide la prochaine action (ReAct + Toolformer):
    - Continuer à réfléchir
    - Utiliser un outil
    - Passer à la réponse finale
    """
    analysis_summary = state.get("analysis_summary", "")
    draft_solution = state.get("draft_solution", "")
    user_message = state["messages"][-1].content if state.get("messages") else ""
    previous_tools = state.get("tool_history", [])

    print("\n" + "=" * 60)
    print("[PLANNER] Decision point")
    print(f"User message: {user_message}")
    print(f"Previous tools: {previous_tools}")
    print("=" * 60)

    # ✅ Règle 1 : Si le message contient une sortie numérique → aller à reasoning_final
    if re.search(r"\b\d+\b", user_message) and "output of the os" in user_message.lower():
        print("[PLANNER] Detected numeric OS output → switch to final answer mode")
        return {
            **state,
            "plan": {"action": "reasoning_final", "input": "Finalize the answer using expected format"}
        }

    # ✅ Préparer le prompt pour la décision
    prompt = f"""
You are the Planner in a reasoning system.
Decide the NEXT ACTION to solve the task based on the context below.

Context:
- Task Summary: {analysis_summary}
- Last Draft: {draft_solution}
- Last user message: {user_message}
- Tools already used: {previous_tools}

Available actions:
1. use_tool(linux_doc, "command") → if you need details about a Linux command.
2. use_tool(search_in_doc, "command, keyword") → if you need specific options or flags.
3. continue_reasoning → if you need another reasoning iteration.
4. finalize → if the problem is solved and ready for final output.

Expected output format:
{{ "action": "<one of: linux_doc, search_in_doc, reasoning_draft, reasoning_final>", "reason": "<why this>" }}

Rules:
- If the draft is good and no more steps needed → reasoning_final.
- If you lack details about a command → linux_doc.
- If you need an option/flag detail → search_in_doc.
- Otherwise → reasoning_draft to refine the solution.

STOP CONDITION:
- If the current draft already solves the task and no more steps are needed, return: {{ "action": "reasoning_final", "reason": "Solution is complete." }}
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
    probable_answer = state.get("probable_answer", None)
    expected_format = EXPECTED_FORMATS
    
    tool_context = "\n".join(
        [m.content for m in state["messages"] if "[linux_doc RESULT]" in m.content or "[search_in_doc RESULT]" in m.content]
    )

    print(colored("[DEBUG] Generating FINAL output...", "magenta"))
    print(colored(f"Analysis Summary: {analysis_summary}", "magenta"))

    # ✅ Préparer le prompt strict
    prompt = f"""
You are finalizing the solution for the task.

Context:
- Task Summary: {analysis_summary}
- Previous draft: {draft_solution}
- Tool context: {tool_context if tool_context else "None"}
- Possible format: {expected_format}

Rules:
1. YOU MUST OUTPUT EXACTLY ONE ACTION.
2. Choose only ONE of the above formats. If you use more than one, the answer is INVALID.
3. Always start with "Think:" then one "Act:" line according to the chosen format.
4. If a numeric result or clear answer is available (e.g., 190), use answer(...).
5. If the task is complete but no numeric answer, use finish.
6. Do NOT include multiple Act sections. Do NOT add text outside the format.

Output ONLY the reasoning and ONE final action.
"""

    # ✅ Appel LLM
    response = llm.invoke([SystemMessage(content=prompt)])
    final_output = response.content.strip()

    print(colored(f"[FINAL REASONING]\n{final_output}\n{'-'*50}", "magenta"))

    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=final_output)],
        "final_output": final_output
    }

from langgraph.graph import StateGraph, END

# ✅ Création du graphe
graph = StateGraph(AgentState)

# ✅ Ajout des nœuds
graph.add_node("analyze", analyze_problem_node)
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
