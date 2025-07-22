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
    expected_format: str
    analysis_summary: str
    draft_solution: str
    tool_context: str
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

import re
from langchain_core.messages import HumanMessage
from termcolor import colored

def linux_doc_node(state: AgentState) -> AgentState:
    # Récupérer le plan
    plan = state.get("plan", {})
    plan_input = plan.get("input", "")

    # Essayer d'extraire le "command" depuis la chaîne
    match = re.search(r'"command"\s*:\s*"([^"]+)"', plan_input)
    command = match.group(1) if match else "ls"  # fallback par défaut

    print(colored(f"[TOOL CALL] linux_doc with command='{command}'", "cyan"))

    # ✅ Exécution du tool
    result = linux_doc(command)
    print(colored(f"[TOOL RESULT] linux_doc returned {len(result)} chars", "cyan"))

    # ✅ Mise à jour de l'état
    new_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=f"[linux_doc RESULT]\n{result}")],
        "tool_history": state.get("tool_history", []) + ["linux_doc"],
        "cycles": state.get("cycles", 0) + 1
    }

    print(colored(f"[STATE AFTER linux_doc_node] Keys: {list(new_state.keys())}", "red"))
    return new_state

def search_in_doc_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    plan_input = plan.get("input", "")

    # Extraire commande et mot-clé via regex
    cmd_match = re.search(r'"command"\s*:\s*"([^"]+)"', plan_input)
    kw_match = re.search(r'"keyword"\s*:\s*"([^"]+)"', plan_input)

    cmd = cmd_match.group(1) if cmd_match else "ls"
    kw = kw_match.group(1) if kw_match else "--help"

    print(colored(f"[TOOL CALL] search_in_doc {cmd}:{kw}", "cyan"))

    # ✅ Exécution du tool
    result = search_in_doc(cmd, kw)

    # ✅ Mise à jour de l'état
    new_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=f"[search_in_doc RESULT]\n{result}")],
        "tool_history": state.get("tool_history", []) + ["search_in_doc"],
        "cycles": state.get("cycles", 0) + 1
    }

    print(colored(f"[STATE AFTER search_in_doc_node] Keys: {list(new_state.keys())}", "cyan"))
    return new_state
