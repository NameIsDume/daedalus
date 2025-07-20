from prompt_and_format import system_prompt
from agent import create_agent, ROLE_MAP
from logger import Logger

from langchain_core.messages import SystemMessage, AIMessage
from termcolor import colored
from hashlib import md5
import asyncio
import httpx
import time

INITIAL_PROMPT_SIGNAL = "You are an assistant that will act like a person"

agents = {}  # {thread_id: {"agent": agent, "memory": memory, "last_used": timestamp}}
seen_messages = set()
# logger = Logger(mode="clean")

def is_duplicate(message: str) -> bool:
    h = md5(message.strip().encode()).hexdigest()
    if h in seen_messages:
        return True
    seen_messages.add(h)
    return False


logger = Logger(verbose=False)  # passe True pour voir tous les détails

def print_message(msg, thread_id=None):
    pass
    # logger.log_message(msg, thread_id)

# logger = Logger(verbose=False)  # ⬅ Mode clean (mettre True pour debug)

# def print_message(msg, thread_id=None):
#     msg_type = type(msg).__name__
#     role = msg_type.replace("Message", "").upper()

#     # Détection des types connus
#     if role in ["SYSTEM", "HUMAN", "AI"]:
#         if role == "HUMAN":
#             role = "USER"
#         elif role == "AI":
#             role = "AGENT"
#     else:
#         role = "MSG"  # fallback pour tool/meta

#     logger.log(role, msg.content, thread_id)

    # #pass
    # msg_type = type(msg).__name__
    # prefix = {
    #     "SystemMessage": colored("[SYS]", "cyan"),
    #     "HumanMessage": colored("[USER]", "green"),
    #     "AIMessage": colored("[AGENT]", "yellow"),
    # }.get(msg_type, "[MSG]")

    # # Récupère les flags du thread
    # flags = agents.get(thread_id, {}).get("log_flags", {})

    # # Filtrage des répétitions
    # if msg_type == "SystemMessage":
    #     if flags.get("sys_printed"):
    #         return  # déjà imprimé
    #     flags["sys_printed"] = True

    # elif msg_type == "AIMessage":
    #     if msg.content.strip() == flags.get("last_ai_msg", ""):
    #         return  # même réponse que la précédente
    #     flags["last_ai_msg"] = msg.content.strip()

    # print(f"{prefix} {msg.content.strip()}")

async def handle_request(thread_id: str, payload: dict) -> dict:
    first_message = payload.get("messages", [{}])[0].get("content", "")
    
    if INITIAL_PROMPT_SIGNAL in first_message:
        if thread_id in agents:
            del agents[thread_id]
            print(colored("[BACKEND]", "red"), f"Hard reset thread: {thread_id}")

    # Trouver ou créer agent
    if thread_id not in agents:
        agent, memory = create_agent()
        agents[thread_id] = {
            "agent": agent,
            "memory": memory,
            "last_used": time.time(),
            "log_flags": {"sys_printed": False, "last_ai_msg": ""}
        }
    else:
        agent = agents[thread_id]["agent"]
        memory = agents[thread_id]["memory"]
        agents[thread_id]["last_used"] = time.time()
        

    return await process_agent_request(agent, memory, payload, thread_id)

async def process_agent_request(agent, memory, payload, thread_id: str) -> dict:
    messages = payload.get("messages", [])
    formatted_messages = [
        ROLE_MAP[m["role"]](content=m["content"]) for m in messages if m["role"] in ROLE_MAP
    ]
    if not any(isinstance(m, SystemMessage) for m in formatted_messages):
        formatted_messages.insert(0, SystemMessage(content=system_prompt))

    config = {"configurable": {"thread_id": thread_id}}
    full_response = {"content": "[ERROR] No response."}

    # Ajout : ID unique pour cette génération (nécessaire pour /api/stop)
    generation_id = f"gen-{thread_id}"

    def run_agent_sync():
        nonlocal full_response
        for step in agent.stream(
            {"messages": formatted_messages},
            config=config,
            stream_mode="values"
        ):
            for msg in step["messages"]:
                print_message(msg, thread_id)
                if isinstance(msg, AIMessage):
                    full_response = {"content": msg.content.strip()}
                    if "act: finish" in msg.content.lower():
                        asyncio.run(memory.adelete_thread(thread_id))

    try:
        # ✅ Lancer la génération dans un thread avec timeout
        await asyncio.wait_for(asyncio.to_thread(run_agent_sync), timeout=60)
    except asyncio.TimeoutError:
        print(colored("[BACKEND]", "red"), f"Timeout after 60s for thread {thread_id}. We need to stop the generation.")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "[ERROR] Response took too long (timeout after 60s). Ollama generation stopped."
                    }
                }
            ]
        }

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": full_response["content"]
                }
            }
        ]
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
