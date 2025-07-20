import re

system_prompt = (
        "You are an autonomous agent operating in a simulated Ubuntu terminal environment. "
    "Your goal is to complete user instructions efficiently and precisely using available shell tools. "
    "For each task, you must first think briefly about what to do, then take exactly one action.\n\n"
    "Guidelines:\n"
    "- Do not speculate or provide commentary.\n"
    "- Keep reasoning minimal and strictly focused on task execution.\n"
    "- Never perform multiple actions in one turn.\n"
    "- Avoid interactive commands (no user input).\n"
    "- Respond only with the required format—no extra text.\n"
)

def remove_multiline_think_blocks(text: str) -> str:
    # Supprime tout ce qui est entre <think>...</think>, même sur plusieurs lignes
    # print("############")
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

# from tools import tools_list

# def my_pre_model_hook(state):
#     last_msg = state["messages"][-1].content.lower()
#     if "paris" in last_msg:
#         raise ValueError("Query blocked: forbidden keyword 'paris'")
#     return state
