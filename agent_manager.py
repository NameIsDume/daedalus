from prompt_and_format import system_prompt
from agent import create_agent, ROLE_MAP

from langchain_core.messages import SystemMessage, AIMessage
from termcolor import colored
from hashlib import md5
import asyncio

import time
agents = {}  # {thread_id: {"agent": agent, "memory": memory, "last_used": timestamp}}
seen_messages = set()

def is_duplicate(message: str) -> bool:
    h = md5(message.strip().encode()).hexdigest()
    if h in seen_messages:
        return True
    seen_messages.add(h)
    return False

def print_message(msg):
    pass
    # msg_type = type(msg).__name__
    # prefix = {
    #     "SystemMessage": colored("[SYS]", "cyan"),
    #     "HumanMessage": colored("[USER]", "green"),
    #     "AIMessage": colored("[AGENT]", "yellow"),
    # }.get(msg_type, "[MSG]")
    # print(f"{prefix} {msg.content.strip()}")

async def handle_request(thread_id: str, payload: dict) -> dict:
    # Trouver ou créer agent
    if thread_id not in agents:
        agent, memory = create_agent()
        agents[thread_id] = {"agent": agent, "memory": memory, "last_used": time.time()}
    else:
        agent = agents[thread_id]["agent"]
        memory = agents[thread_id]["memory"]
        agents[thread_id]["last_used"] = time.time()

    # agent, memory = agents[thread_id]

    # Si doublon → nouveau thread_id
    message = payload.get("prompt", "")
    if is_duplicate(message):
        prefix = colored("[BACKEND]", "red", attrs=["bold"])
        print(f"{prefix} Duplicate message detected, creating new thread_id for: {message}")
        thread_id += "_dup"
        agent, memory = create_agent()
        agents[thread_id] = {"agent": agent, "memory": memory, "last_used": time.time()}

    return await process_agent_request(agent, memory, payload, thread_id)

import json

async def process_agent_request(agent, memory, payload, thread_id: str) -> dict:
    messages = payload.get("messages", [])
    formatted_messages = [
        ROLE_MAP[m["role"]](content=m["content"]) for m in messages if m["role"] in ROLE_MAP
    ]
    if not any(isinstance(m, SystemMessage) for m in formatted_messages):
        formatted_messages.insert(0, SystemMessage(content=system_prompt))
    config = {"configurable": {"thread_id": thread_id}}
    full_response = None

    print(colored("[BACKEND]", "red"), "Sending to Ollama:")
    print(json.dumps({"messages": [m.content for m in formatted_messages]}, indent=2))
    for step in agent.stream({"messages": formatted_messages}, config=config, stream_mode="values"):
        for msg in step["messages"]:
            print_message(msg)
            if isinstance(msg, AIMessage):
                full_response = msg
                if "act: finish" in msg.content.lower():
                    await memory.adelete_thread(thread_id)
                    del agents[thread_id]

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": full_response.content.strip() if full_response else ""
            }
        }]
    }

def get_memory(thread_id):
    if thread_id in agents:
        _, memory = agents[thread_id]
        return memory.get({"configurable": {"thread_id": thread_id}})
    return None

async def cleanup_inactive_threads(ttl: int = 10, interval: int = 5):
    """
    Supprime les threads inactifs depuis plus de `ttl` secondes.
    Vérifie toutes les `interval` secondes.
    """
    while True:
        now = time.time()
        prefix = colored("[BACKEND]", "red", attrs=["bold"])
        to_delete = [tid for tid, data in agents.items() if now - data["last_used"] > ttl]
        for tid in to_delete:
            del agents[tid]
            print(f"{prefix} Removed inactive thread: {tid}")
        await asyncio.sleep(interval)
