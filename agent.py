from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from model import model
import re
# from tools import tools_list

def remove_multiline_think_blocks(text: str) -> str:
    # Supprime tout ce qui est entre <think>...</think>, même sur plusieurs lignes
    print("############")
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

app = FastAPI()
memory = MemorySaver()
# def my_pre_model_hook(state):
#     last_msg = state["messages"][-1].content.lower()
#     if "paris" in last_msg:
#         raise ValueError("Query blocked: forbidden keyword 'paris'")
#     return state

agent = create_react_agent(
    model=model,
    # tools=tools_list,
    tools=[],
    checkpointer=memory,
    # pre_model_hook=my_pre_model_hook,
)

system_prompt = (
    "You are an autonomous agent designed to complete user instructions efficiently and precisely. "
    "Always follow the requested format and focus on solving the task, not explaining it unnecessarily. "
    "Keep your reasoning minimal and relevant. Avoid verbosity and speculative thinking. "
    "Do not overthink—act with clarity and purpose."
)

class AgentInput(BaseModel):
    prompt: str

import json

import termcolor
from termcolor import colored

def print_message(msg):
    msg_type = type(msg).__name__
    prefix = {
        "SystemMessage": colored("[SYS]", "cyan"),
        "HumanMessage": colored("[USER]", "green"),
        "AIMessage": colored("[AGENT]", "yellow"),
    }.get(msg_type, "[MSG]")
    print(f"{prefix} {msg.content.strip()}")

@app.post("/api/chat")
async def run_agent_ollama_format(request: Request):

    payload = await request.json()
    messages = payload.get("messages", [])
    # model_name = payload.get("model", "unknown")
    # stream = payload.get("stream", False)

    # print(f"Received prompt: {messages}")
    # print(json.dumps(messages, indent=2))
    # print(f"Model: {model_name}")

    # for m in messages:
    #     print(f"- {m.get('role', '?')}: {m.get('content', '')[:100]}")

    formatted_messages: list[BaseMessage] = [
        ROLE_MAP[m["role"]](content=m["content"])
        for m in messages
        if m["role"] in ROLE_MAP
    ]

    if not any(isinstance(m, SystemMessage) for m in formatted_messages):
        formatted_messages.insert(0, SystemMessage(content=system_prompt))

    config = {"configurable": {"thread_id": "default"}}

    try:
        full_response = ""
        for i, step in enumerate(agent.stream({"messages": formatted_messages}, config=config, stream_mode="values")):
            for msg in step["messages"]:
                print_message(msg)
            if isinstance(msg, AIMessage):
                full_response += msg.content + "\n"
        # for step in agent.stream({"messages": formatted_messages}, config=config, stream_mode="values"):
        #     last_msg = step["messages"][-1]
        #     if isinstance(last_msg, AIMessage):
        #         full_response += last_msg.content + "\n"

        print("#### Final Response ####")
        response_text = remove_multiline_think_blocks(full_response.strip())
        print(f"{response_text}\n")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    }
                }
            ]
        }
    except ValueError as e:
        return {
            "choices": [
                {
                    "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                    }
                }
            ]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)