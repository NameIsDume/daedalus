from model import model_llm
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from termcolor import colored
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[BaseMessage]
    expected_format: str
    analysis_summary: str
    current_problem: str
    last_action: str
    draft_solution: str
    tool_context: str
    cycles: int

llm = model_llm

def reasoning_draft_first_interaction(state: AgentState) -> AgentState:
    """
    First reasoning step for the initial user interaction.
    """
    current_problem = state.get("current_problem", "")

    print(colored("[DEBUG] First interaction reasoning...", "yellow"))
    print(colored(f"Current Problem: {current_problem}", "yellow"))
    prompt = f"""
    You are an assistant that will act like a person. You MUST follow a strict process to complete the task.
    RULES:
    - You always output in the format:
    Think: <your reasoning>
    Act: ```bash\n
    # put your bash code here if needed\n```
    NEVER output explanations or multiple actions.
    
    Current Problem: {current_problem}
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    draft_solution = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_solution}\n{'-'*50}", "red"))

    return {
        **state,
        "draft_solution": draft_solution,
        "messages": state["messages"] + [HumanMessage(content=f"[Draft Solution]\n{draft_solution}")]
    }
    
def reasoning_draft_multiple_steps(state: AgentState) -> AgentState:
    current_problem = state.get("current_problem", "")
    analysis_summary = state.get("analysis_summary", "No summary available.")
    last_action = state.get("last_action", None)
    
    print(colored("[DEBUG] Generating draft reasoning for multiple steps...", "yellow"))
    print(colored(f"Last Action: {last_action}", "yellow"))

def reasoning_draft_node(state: AgentState) -> AgentState:
    last_action = state.get("last_action", None)
    if last_action == None:
        print(colored("[DEBUG] No last action found, using initial reasoning.", "yellow"))
        return reasoning_draft_first_interaction(state)
    else:
        print(colored("[DEBUG] Continuing with multi-step reasoning...", "yellow"))
        return reasoning_draft_multiple_steps(state)
    
# def reasoning_draft_node(state: AgentState) -> AgentState:
#     analysis_summary = state.get("analysis_summary", "No summary available.")
#     # user_message = state["messages"][-1].content if state.get("messages") else ""
#     current_problem = state.get("current_problem", "")
#     last_action = state.get("last_action", None)

#     print(colored("[DEBUG] Generating draft reasoning...", "yellow"))
#     print(colored(f"Analysis Summary: {analysis_summary}", "yellow"))
#     print(colored(f"Current Problem: {current_problem}", "yellow"))
#     # print(colored(f"Last User Message: {user_message}", "yellow"))

#     prompt = f"""
# You are an assistant that will act like a person. You MUST follow a strict multi-step process to complete the task.

# RULES:
# - You always output in the format:
# Think: <your reasoning>
# Act: <one of bash, answer, or finish>
# - ONE action per step.
# - bash → execute command.
# - answer(<value>) → when you have the requested result.
# - finish → ONLY when user confirms the task is done.
# - NEVER output explanations or multiple actions.
# - Use previous steps for reasoning, do not repeat them.

# Current Problem: {current_problem}
# Current Analysis Summary: {analysis_summary}
# Last time you :
# {last_action}
# """
# # Current User Message: {user_message}

#     # ✅ Appel LLM
#     response = llm.invoke([SystemMessage(content=prompt)])
#     draft_solution = response.content.strip()

#     print(colored(f"[DRAFT REASONING]\n{draft_solution}\n{'-'*50}", "red"))

#     return {
#         **state,
#         "draft_solution": draft_solution,
#         "messages": state["messages"] + [HumanMessage(content=f"[Draft Solution]\n{draft_solution}")]
#     }