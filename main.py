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

MAX_CYCLES = 2
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
    draft_solution: str
    cycles: int

class ChatInput(BaseModel):
    prompt: str
    thread_id: str = "default"

# ==========================================
# 3. Initialiser le modèle
# ==========================================
llm = model_llm  # Remplace par ton modèle local (ex: Ollama)


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

def extract_format_node(state: AgentState) -> AgentState:
    """
    Extrait le format attendu du premier message utilisateur (benchmark instruction).
    """
    user_instruction = state["messages"][0].content

    prompt = f"""
You are a strict extractor. The user provided an instruction for how the output should be formatted.
Extract ONLY the output format from the following text:
---
{user_instruction}
---
Return ONLY the template of the expected format (no explanations).
"""

    response = llm.invoke([SystemMessage(content=prompt)])
    extracted_format = response.content.strip()

    print(colored(f"[DEBUG] Extracted Expected Format:\n{extracted_format}\n{'-'*50}", "blue"))

    return {
        "messages": state["messages"] + [HumanMessage(content=f"[Expected Format Extracted]\n{extracted_format}")],
        "expected_format": extracted_format
    }

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

def reasoning_draft_node(state: AgentState) -> AgentState:
    user_request = state["messages"][-1].content
    analysis_summary = state.get("analysis_summary", "No summary.")

    prompt = f"""
You are a Linux command generator. The user wants: "{user_request}"
Analysis Summary: "{analysis_summary}"

Generate the BEST possible bash command to achieve this.
Follow format:
Think: (why this command solves it)
Bash: (the exact command)
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    draft_output = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_output}\n{'-'*50}", "cyan"))
    return {
        "draft_solution": draft_output,
        "messages": state["messages"] + [HumanMessage(content=f"[DRAFT]\n{draft_output}")]
    }

def planner_node(state: AgentState) -> AgentState:
    draft = state.get("draft_solution", "")
    tool_history = state.get("tool_history", [])
    analysis_summary = state.get("analysis_summary", "")

    prompt = f"""
You are the Planner. Decide NEXT action.

User task: "{analysis_summary}"
Draft solution: {draft}

Tool history: {tool_history if tool_history else "None"}

Rules:
- If the draft solves the task → finish
- Else:
  - If linux_doc not used → linux_doc
  - Else → search_in_doc
Return ONLY JSON in one of these formats:
{{"action": "linux_doc", "input": {{"command": "<command>"}}}}
{{"action": "search_in_doc", "input": {{"command": "<command>", "keyword": "<keyword>"}}}}
{{"action": "finish", "input": {{"answer": "<answer>"}}}}
"""

    response = llm.invoke([SystemMessage(content=prompt)])
    raw_output = response.content.strip()
    print(colored(f"[PLANNER RAW]\n{raw_output}\n{'-'*50}", "yellow"))

    try:
        json_text = extract_json(raw_output)
        parsed = PlanOutput.model_validate_json(json_text)
    except Exception as e:
        print(colored(f"[WARN] Fallback: invalid JSON → finish", "red"))
        parsed = PlanOutput(action="finish", input={"answer": draft})

    return {
        "plan": parsed,
        "messages": state["messages"] + [HumanMessage(content=f"[PLANNER]\n{parsed.model_dump_json()}")]
    }


# ==========================================
# Analyze node
# ==========================================
from langchain_core.messages import SystemMessage, HumanMessage
from termcolor import colored

def reasoning_final_node(state: AgentState) -> AgentState:
    analysis_summary = state.get("analysis_summary", "No summary.")
    draft_solution = state.get("draft_solution", "")
    tool_context = "\n".join(
        [m.content for m in state["messages"] if "[linux_doc RESULT]" in m.content or "[search_in_doc RESULT]" in m.content]
    )

    prompt = f"""
Refine the previous solution using additional context.

User goal: "{analysis_summary}"
Previous draft:
{draft_solution}

Tool context:
{tool_context if tool_context else "No extra info."}

Generate improved version:
Think: (reasoning)
Bash: (final command)
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    final_output = response.content.strip()

    print(colored(f"[FINAL REASONING]\n{final_output}\n{'-'*50}", "magenta"))
    return {
        "messages": state["messages"] + [HumanMessage(content=final_output)]
    }

def formatter_node(state: AgentState) -> AgentState:
    raw_reasoning = state["messages"][-1].content
    expected_format = state.get("expected_format", "Think: ...\n\nAct: bash\n\n```bash\n<command>\n```")

    prompt = f"""
Take the reasoning below and output ONLY in the expected format.

Reasoning:
---
{raw_reasoning}
---

Expected format:
---
{expected_format}
---

DO NOT include any explanations. Output ONLY the formatted response.
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    formatted_output = response.content.strip()

    print(colored(f"[FINAL OUTPUT]\n{formatted_output}\n{'-'*50}", "green"))
    return {
        "messages": state["messages"] + [HumanMessage(content=formatted_output)]
    }


from termcolor import colored
from langchain_core.messages import HumanMessage
from typing import Dict

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

# def extract_json(text: str) -> str:
#     """Extrait le premier objet JSON valide trouvé dans un texte."""
#     match = re.search(r"\{[\s\S]*\}", text)
#     if match:
#         return match.group(0)
#     raise ValueError("No JSON found")

# ==========================================
# 6. Construire le Graph LangGraph
# ==========================================
graph = StateGraph(AgentState)

def route_from_plan(state: AgentState) -> str:
    action = state["plan"].action
    cycles = state.get("cycles", 0)
    history = state.get("tool_history", [])

    # Si déjà trop de cycles → on force reasoning_final
    if cycles >= MAX_CYCLES:
        print(colored(f"[WARN] Max cycles reached ({cycles}). Going to reasoning_final.", "red", attrs=["bold"]))
        return "reasoning_final"

    # Si finish → on saute aux reasoning_final
    if action == "finish":
        return "reasoning_final"

    # Sinon, plan normal
    if action == "linux_doc":
        return "linux_doc"
    elif action == "search_in_doc":
        return "search_in_doc"

    return "reasoning_final"

graph = StateGraph(AgentState)

graph.add_node("extract_format", extract_format_node)
graph.add_node("analyze", analyze_task)
graph.add_node("reasoning_draft", reasoning_draft_node)
graph.add_node("planner", planner_node)
graph.add_node("linux_doc", linux_doc_node)
graph.add_node("search_in_doc", search_in_doc_node)
graph.add_node("reasoning_final", reasoning_final_node)
graph.add_node("formatter", formatter_node)

graph.add_edge("extract_format", "analyze")
graph.add_edge("analyze", "reasoning_draft")
graph.add_edge("reasoning_draft", "planner")

graph.add_conditional_edges("planner", route_from_plan, {
    "linux_doc": "linux_doc",
    "search_in_doc": "search_in_doc",
    "reasoning_final": "reasoning_final"
})

graph.add_edge("linux_doc", "planner")
graph.add_edge("search_in_doc", "planner")
graph.add_edge("reasoning_final", "formatter")
graph.add_edge("formatter", END)

graph.set_entry_point("extract_format")


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