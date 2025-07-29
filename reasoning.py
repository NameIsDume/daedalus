from model import model_llm
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from termcolor import colored
from typing import TypedDict, List
import re

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
    You are an assistant that will act like a person. You MUST follow a strict multi-step process to complete the task.

    RULES:
    - You MUST choose EXACTLY ONE of the following action formats at the end of your response:

    1. To execute a bash command:
    Think: <your reasoning>
    Act: bash

    ```bash
    # put your bash code here if needed\n```
    NEVER output explanations or multiple actions.

    Current Problem: {current_problem}
    """
    # prompt = f"""
    # You are an assistant that will act like a person. You MUST follow a strict process to complete the task.
    # RULES:
    # - You always output in the format:
    # Think: <your reasoning>
    # Act: bash\n```bash\n
    # # put your bash code here if needed\n```
    # NEVER output explanations or multiple actions.

    # Current Problem: {current_problem}
    # """
    response = model_llm.invoke([SystemMessage(content=prompt)])
    draft_solution = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_solution}\n{'-'*50}", "red"))

    return {
        **state,
        "draft_solution": draft_solution,
    }

def reasoning_draft_multiple_steps(state: AgentState) -> AgentState:
    current_problem = state.get("current_problem", "")
    analysis_summary = state.get("analysis_summary", "No summary available.")
    last_action = state.get("last_action", None)

    print(colored("[DEBUG] Generating draft reasoning for multiple steps...", "yellow"))
    print(colored(f"Last Action: {last_action}", "yellow"))
    print(colored(f"Current Problem: {current_problem}", "yellow"))
    print(colored(f"Analysis Summary: {analysis_summary}", "yellow"))
    messages = state.get("messages", [])

    print("Les clés du state:", state.keys())

    print(colored("[DEBUG] Messages in state:", "cyan"))
    print(messages)

    match = re.search(r"The output of the OS:\s*(\d+)", messages[-1].content)
    if match and last_action and "```bash" in last_action:
        value = match.group(1)
        draft_solution = f"Think: The last command returned a numeric value, which likely answers the question directly.\n\nAct: answer({value})"
        print(colored(f"[DRAFT REASONING - SHORTCUT]\n{draft_solution}\n{'-'*50}", "green"))
        return {
            **state,
            "draft_solution": draft_solution,
        }

    prompt = f"""
    You are an assistant that will act like a person. You MUST follow a strict multi-step process to complete the task.
    Current Problem: {current_problem}
    The last action you took was: {last_action}
    The output you received from the OS was: {analysis_summary}
    Is the output you received from the OS the answer you need to give ?
    If it is, you should output the answer in the format:
    Think: <your reasoning>
    Act: answer(<value>)
    """

    response = model_llm.invoke([SystemMessage(content=prompt)])
    draft_solution = response.content.strip()

    print(colored(f"[DRAFT REASONING]\n{draft_solution}\n{'-'*50}", "red"))
    return {
        **state,
        "draft_solution": draft_solution,
    }

def reasoning_draft_node(state: AgentState) -> AgentState:
    last_action = state.get("last_action", None)
    print(colored("[DEBUG] Checking last action...", "yellow"))
    print(colored(f"Last Action: {last_action}", "yellow"))

    if not last_action:
        print(colored("[DEBUG] No last action found, using initial reasoning.", "yellow"))
        return reasoning_draft_first_interaction(state)
    else:
        print(colored("[DEBUG] Continuing with multi-step reasoning...", "yellow"))
        return reasoning_draft_multiple_steps(state)
