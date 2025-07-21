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
    command = state["plan"].input.get("command")
    print(colored(f"[TOOL CALL] linux_doc with command='{command}'", "yellow", attrs=["bold"]))

    try:
        result = linux_doc.run(command)
        print(colored(f"[TOOL RESULT] linux_doc returned {len(result)} chars", "cyan"))

        return {"messages": state["messages"] + [HumanMessage(content=f"[linux_doc RESULT]\n{result[:200]}...")]}
    except Exception as e:
        print(colored(f"[ERROR] linux_doc failed: {str(e)}", "red", attrs=["bold"]))
        return {"messages": state["messages"] + [HumanMessage(content=f"[linux_doc ERROR]\n{str(e)}")]}

def search_in_doc_node(state: AgentState) -> AgentState:
    command = state["plan"].input.get("command")
    keyword = state["plan"].input.get("keyword")

    print(colored(f"[TOOL CALL] search_in_doc with command='{command}', keyword='{keyword}'", "yellow", attrs=["bold"]))

    try:
        result = search_in_doc.run({"command": command, "keyword": keyword})
        print(colored(f"[TOOL RESULT] search_in_doc returned {len(result.splitlines())} lines", "cyan"))

        return {"messages": state["messages"] + [HumanMessage(content=f"[search_in_doc RESULT]\n{result[:200]}...")]}

    except Exception as e:
        print(colored(f"[ERROR] search_in_doc failed: {str(e)}", "red", attrs=["bold"]))
        return {"messages": state["messages"] + [HumanMessage(content=f"[search_in_doc ERROR]\n{str(e)}")]}
