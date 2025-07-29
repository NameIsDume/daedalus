from fastapi import FastAPI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional, TypedDict, List
from langchain_core.messages import BaseMessage
from termcolor import colored

import uvicorn
import httpx

from model import model_llm
from tools import linux_doc_node, search_in_doc_node
from reasoning import reasoning_draft_node
from analyse import analyse_problem_node

MAX_CYCLES = 2

llm = model_llm

class FinalResponse(BaseModel):
    """Structured response for final reasoning output"""
    thought: str = Field(description="The final reasoning or explanation.")
    action: str = Field(description="The final action to take. Must be one of: bash, answer(...), or finish.")
    code: str = Field(default="", description="If action is bash, the bash command to run.")

class AgentState(TypedDict):
    messages: List[BaseMessage]
    expected_format: str
    analysis_summary: str
    current_problem: str
    last_action: str
    draft_solution: str
    tool_context: str
    cycles: int

model_with_structured_output = llm.with_structured_output(FinalResponse)

import re

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
    previous_tools = state.get("tool_history", [])

    print("\n" + "=" * 60)
    print("[PLANNER] Decision point")
    print("=" * 60)

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
    elif re.search(r"^Act:\s*bash\s*$\n+```bash", draft_solution, re.MULTILINE):
        print("[PLANNER] Detected bash command → switch to reasoning_final")
        return {
            **state,
            "plan": {"action": "reasoning_final", "input": "bash command detected"}
        }

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

def reasoning_final_node(state: AgentState):
    current_problem = state.get("current_problem", "")
    output_os = state.get("output_of_os", "")
    reasoning = state.get("draft_solution", "")

    print("\n" + "=" * 60)
    print("[FINAL REASONING] Finalizing the task")
    print(f"Current Problem: {current_problem}")
    print(f"Previous Output: {output_os}")
    print(f"Reasoning: {reasoning}")

    # Here we generated a structured output that matche the benchmark expectation
    structured = model_with_structured_output.invoke([
        SystemMessage(content="You are finalizing the task. Output must follow structured reasoning format."),
        HumanMessage(content=f"Task: {current_problem}\nPrevious Output: {output_os}\nReasoning: {reasoning}")
    ])

    formatted_msg = f"Think: {structured.thought}\nAct: {structured.action}\n{structured.code}"
    print(f"FORMATED MESSAGE: {formatted_msg}")
    print(colored(f"[FINAL REASONING]\n{structured}\n{'-'*50}", "magenta"))
    print("FINAL STRUCTURED OUTPUT")
    # final_str = f"Think: {structured.thought}\nAct: {structured.action}"
    # print(final_str)

    if structured.action.strip() == "bash":
        final_str = f"Think: {structured.thought}\nAct: bash\n\n```bash\n{structured.code.strip()}\n```"
    elif structured.action.startswith("answer("):
        final_str = f"Think: {structured.thought}\nAct: {structured.action}"
    elif structured.action.strip() == "finish":
        final_str = f"Think: {structured.thought}\nAct: finish"
    else:
        raise ValueError(f"Invalid action returned: {structured.action}")

    print(final_str)

    return {
        **state,
        "final_response": final_str,
        "last_action": final_str,
    }

from langgraph.graph import StateGraph, END


graph = StateGraph(AgentState)


graph.add_node("analyze", analyse_problem_node)
graph.add_node("reasoning_draft", reasoning_draft_node)
graph.add_node("planner", planner_node)
graph.add_node("linux_doc", linux_doc_node)
graph.add_node("search_in_doc", search_in_doc_node)
graph.add_node("reasoning_final", reasoning_final_node)

graph.add_edge("analyze", "reasoning_draft")
graph.add_edge("reasoning_draft", "planner")

graph.add_conditional_edges("planner", lambda state: state["plan"]["action"], {
    "linux_doc": "linux_doc",
    "search_in_doc": "search_in_doc",
    "reasoning_draft": "reasoning_draft",
    "reasoning_final": "reasoning_final"
})

graph.add_edge("linux_doc", "planner")
graph.add_edge("search_in_doc", "planner")
graph.add_edge("reasoning_final", END)

graph.set_entry_point("analyze")


from routes import create_router
checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

# Crée le router avec le graph
router = create_router(app_graph)

app = FastAPI(title="Linux Agent API")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=11435, reload=True)
