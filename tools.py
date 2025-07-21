from langchain_core.tools import tool
from termcolor import colored
import httpx
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from typing import Literal, Dict
from langchain_core.messages import HumanMessage

class PlanOutput(BaseModel):
    action: Literal["linux_doc", "search_in_doc", "finish"]
    input: Dict[str, str]

class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: PlanOutput
    tool_history: List[str]
    draft_solution: str
    cycles: int


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
    print(colored(f"[TOOL CALL] linux_doc with command='{command}'", "cyan"))
    
    result = linux_doc(command)  # ton implémentation
    print(colored(f"[TOOL RESULT] linux_doc returned {len(result)} chars", "cyan"))
    
    return {
        "messages": state["messages"] + [HumanMessage(content=f"[linux_doc RESULT]\n{result}")],
        "tool_history": state.get("tool_history", []) + ["linux_doc"],
        "cycles": state.get("cycles", 0) + 1
    }

def search_in_doc_node(state: AgentState) -> AgentState:
    cmd = state["plan"].input["command"]
    kw = state["plan"].input["keyword"]
    print(colored(f"[TOOL CALL] search_in_doc {cmd}:{kw}", "cyan"))
    
    result = search_in_doc(cmd, kw)
    
    return {
        "messages": state["messages"] + [HumanMessage(content=f"[search_in_doc RESULT]\n{result}")],
        "tool_history": state.get("tool_history", []) + ["search_in_doc"],
        "cycles": state.get("cycles", 0) + 1
    }
