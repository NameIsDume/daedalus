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
# ==========================================
# 1. Définir les Schemas Pydantic
# ==========================================
class PlanOutput(BaseModel):
    action: Literal["linux_doc", "search_in_doc", "finish"]
    input: Dict[str, str]

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    plan: Optional[PlanOutput]
    tool_history: List[str]
    draft_solution: str
    cycles: int
    analysis_summary: str
    expected_format: str

initial_state: AgentState = {
    "messages": [],
    "tool_history": [],
    "draft_solution": "",
    "cycles": 0,
    "analysis_summary": "",
    "expected_format": ""
}

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
        **state,
        "expected_format": extracted_format,
        "messages": state["messages"] + [HumanMessage(content=f"[Expected Format Extracted]\n{extracted_format}")]
    }

def analyze_task(state: Dict) -> Dict:
    """
    Analyse la consigne utilisateur et produit un résumé simple.
    """
    user_message = state["messages"][0].content
    # print(colored(f"[DEBUG] Analyzing user message: {user_message}", "blue", attrs=["bold"]))

    if "problem is:" in user_message.lower():
        # print(colored(f"[DEBUG] Analysis Summary: Problem detected", "blue", attrs=["bold"]))
        user_message = user_message.split("problem is:")[-1].strip()
        # print(f"Extracted problem description: {user_message}")
    else:
        # print(colored(f"[DEBUG] Analysis Summary: No problem detected", "blue", attrs=["bold"]))
        user_message = state["messages"][-1].content  # Dernier message utilisateur

    prompt = f"""
Summarize the following request in ONE short sentence.
- Do NOT include any code, commands, or solutions.
- Do NOT explain, just give the task in abstract form.

Request:
{user_message}

Return EXACTLY this format:
Summary: <one concise sentence>
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    summary_text = response.content.strip()

    print(colored(f"[DEBUG] Analysis Summary: {summary_text}", "blue", attrs=["bold"]))
    new_state = {
        **state,  # merge l'état précédent
        "analysis_summary": summary_text,
        "messages": state["messages"] + [HumanMessage(content=f"[ANALYSIS]\n{summary_text}")]
    }
    print(colored(f"[STATE AFTER analyze_task] Keys: {list(new_state.keys())}", "red"))
    return new_state

def reasoning_draft_node(state: AgentState) -> AgentState:
    print(colored(f"[STATE DEBUG] Keys: {list(state.keys())}", "red"))
    analysis_summary = state.get("analysis_summary", "No summary.")

    print(colored(f"[DEBUG] Generating draft command for: {analysis_summary}", "blue", attrs=["bold"]))

    prompt = f"""
You are a Linux command generator.
Goal: "{analysis_summary}"

Generate the BEST possible bash command to achieve this goal.
Follow format:
Think: (why this solves the goal)
Bash: (exact command)
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    draft_output = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_output}\n{'-'*50}", "cyan"))

    # ✅ Stocke le nouvel état dans une variable
    new_state = {
        **state,
        "draft_solution": draft_output,
        "messages": state["messages"] + [HumanMessage(content=f"[DRAFT]\n{draft_output}")]
    }

    print(colored(f"[STATE AFTER reasoning_draft_node] Keys: {list(new_state.keys())}", "red"))
    return new_state

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
        **state,
        "messages": state["messages"] + [HumanMessage(content=formatted_output)]
    }

def planner_node(state: AgentState) -> AgentState:
    draft = state.get("draft_solution", "")
    analysis_summary = state.get("analysis_summary", "")
    tool_history = state.get("tool_history", [])

    prompt = f"""
You are the Planner. Decide what to do NEXT.

User request: "{analysis_summary}"
Draft solution:
{draft}

Evaluate first:
1. Does the draft command ALREADY satisfy the user request fully? If yes → finish.
2. If NOT:
   - If linux_doc not used → linux_doc
   - Else → search_in_doc
3. Do NOT choose the same tool twice in a row.

IMPORTANT: Always return ONLY JSON in ONE of these formats:
{{
  "action": "finish",
  "input": {{"answer": "<use the draft solution as answer>"}}
}}
{{
  "action": "linux_doc",
  "input": {{"command": "<command name only, no args>"}}
}}
{{
  "action": "search_in_doc",
  "input": {{"command": "<command>", "keyword": "<keyword>"}}
}}
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    raw_output = response.content.strip()
    print(colored(f"[PLANNER RAW]\n{raw_output}\n{'-'*50}", "yellow"))

    try:
        json_text = extract_json(raw_output)
        parsed = PlanOutput.model_validate_json(json_text)
    except Exception:
        print(colored("[WARN] Planner returned invalid JSON → fallback to finish", "red"))
        parsed = PlanOutput(action="finish", input={"answer": draft})

    return {
        **state,
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
Refine the previous solution so that it COMPLETELY fulfills the user request:
"{analysis_summary}"

Rules:
- If the previous command already works → keep it as is.
- If incomplete → correct it using the context below.
- Use ONLY what is relevant from tool context.

Previous draft:
{draft_solution}

Tool context:
{tool_context if tool_context else "No extra info."}

Return in format:
Think: <reasoning>
Bash: <final command>
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    final_output = response.content.strip()

    print(colored(f"[FINAL REASONING]\n{final_output}\n{'-'*50}", "magenta"))
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=final_output)]
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
