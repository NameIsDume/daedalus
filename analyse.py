from termcolor import colored
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import TypedDict, List
from model import model_llm

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

def analyse_node_first_interaction(state: AgentState) -> AgentState:
    """
    Analyse le premier message utilisateur pour définir le but initial.
    """
    user_message = state["messages"][-1].content if state.get("messages") else ""
    # print(colored(f"[DEBUG] First interaction analysis:\n{user_message}\n{'-'*50}", "cyan"))

    if "problem is" in user_message.lower():
        # Si le message contient "problem is", on en déduit le but initial
        user_message = user_message.split("problem is", 1)[-1].strip()

    prompt = f"""
Summarize the user's problem in one short sentence.
User message:
{user_message}
Rules:
- Do NOT propose a solution.
- Max 30 words.
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    analysis_summary = response.content.strip()
    
    print(colored(f"[DEBUG] Initial Problem Analysis: {analysis_summary}", "red"))
    return {
        **state,
        "analysis_summary": analysis_summary,
        "current_problem": analysis_summary,
        "messages": state["messages"] + [HumanMessage(content=f"[Analysis Summary]\n{analysis_summary}")]
    }
    
def analyse_node_previous_summary(state: AgentState) -> AgentState:
    """
    We already have a previous summary, so we act differently:
    """
    user_message = state["messages"][-1].content if state.get("messages") else ""
    current_problem = state.get("current_problem", "")
    previous_summary = state.get("analysis_summary", "")
    last_action = state.get("last_action", None)
    print(colored(f"[DEBUG] Previous summary analysis:\n{user_message}\n{'-'*50}", "cyan"))

    prompt = f"""
You are analyzing the output of an executed command in a multi-step reasoning process.
The final goal is: "{current_problem}"
Previous summary: "{previous_summary}"
Last executed action: "{last_action}"
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
    
    return {
        **state,
        "analysis_summary": analysis_summary,
        "messages": state["messages"] + [HumanMessage(content=f"[Analysis Summary]\n{analysis_summary}")]
    }

def analyse_problem_node(state: AgentState) -> AgentState:
    """
    If there is no previous summary, we analyze the first interaction.
    If there is a previous summary, we analyze the last user message in context of that summary
    If there is "start" a new problem" in the last message, we reset the analysis.
    """
    previous_summary = state.get("analysis_summary", "")

    if not previous_summary:
        return analyse_node_first_interaction(state)
    else:
        return analyse_node_previous_summary(state)
