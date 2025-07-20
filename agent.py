import prompt_and_format as prompt

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from model import model_llm
from tools import linux_doc

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

memory = MemorySaver()

def post_model_hook(state: dict) -> dict:
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage):
        last_ai = messages[-1]
        cleaned = prompt.remove_multiline_think_blocks(last_ai.content)
        last_ai.content = cleaned
    return state

def create_agent():
    memory = MemorySaver()
    agent = create_react_agent(
        model=model_llm,
        # tools=tools_list,
        tools=[linux_doc],
        checkpointer=memory,
        post_model_hook=post_model_hook,
        # pre_model_hook=my_pre_model_hook,
    )
    return agent, memory

