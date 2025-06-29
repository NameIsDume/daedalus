import asyncio
import uuid
import prompt_and_format as prompt
import termcolor
from termcolor import colored
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from model import model
# from tools import tools_list

# def my_pre_model_hook(state):
#     last_msg = state["messages"][-1].content.lower()
#     if "paris" in last_msg:
#         raise ValueError("Query blocked: forbidden keyword 'paris'")
#     return state

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

app = FastAPI()
# File des requêtes entrantes
task_queue = asyncio.Queue()
NUM_WORKERS = 2

# Statistiques basiques
queue_counter = 0

memory = MemorySaver()

agent = create_react_agent(
    model=model,
    # tools=tools_list,
    tools=[],
    checkpointer=memory,
    # pre_model_hook=my_pre_model_hook,
)

class ChatTask:
    def __init__(self, payload, response_future):
        self.payload = payload
        self.response_future = response_future
        self.id = uuid.uuid4().hex

def print_message(msg):
    msg_type = type(msg).__name__
    prefix = {
        "SystemMessage": colored("[SYS]", "cyan"),
        "HumanMessage": colored("[USER]", "green"),
        "AIMessage": colored("[AGENT]", "yellow"),
    }.get(msg_type, "[MSG]")
    print(f"{prefix} {msg.content.strip()}")

@app.on_event("startup")
async def start_workers():
    print(f"🔧 Launching {NUM_WORKERS} workers...")
    for i in range(NUM_WORKERS):
        asyncio.create_task(agent_worker(i))

@app.get("/api/status")
async def status():
    return {
        "pending_tasks": task_queue.qsize(),
        "max_concurrent": NUM_WORKERS
    }

async def agent_worker(worker_id):
    while True:
        task: ChatTask = await task_queue.get()
        print(f"[🧠 Worker {worker_id}] Processing task {task.id} (queue size: {task_queue.qsize()})")
        try:
            response = await asyncio.wait_for(process_agent_request(task.payload), timeout=110)
            task.response_future.set_result(response)
        except asyncio.TimeoutError:
            print(f"[⏱ Timeout] Task {task.id} exceeded 110s")
            task.response_future.set_result({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Error: request timed out after 110 seconds."
                    }
                }]
            })
        except Exception as e:
            print(f"[💥 Error] Task {task.id}: {str(e)}")
            task.response_future.set_result({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                }]
            })
        task_queue.task_done()

async def process_agent_request(payload):
    messages = payload.get("messages", [])
    formatted_messages: list[BaseMessage] = [
        ROLE_MAP[m["role"]](content=m["content"]) for m in messages if m["role"] in ROLE_MAP
    ]

    if not any(isinstance(m, SystemMessage) for m in formatted_messages):
        formatted_messages.insert(0, SystemMessage(content=prompt.system_prompt))

    config = {"configurable": {"thread_id": "default"}}
    full_response = ""

    for step in agent.stream({"messages": formatted_messages}, config=config, stream_mode="values"):
        for msg in step["messages"]:
            print_message(msg)
        if isinstance(msg, AIMessage):
            full_response += msg.content + "\n"

            if "act: finish" in msg.content.lower():
                print("🧼 Fin détectée, reset de l'historique du thread.")
                memory.delete("default")

    response_text = prompt.remove_multiline_think_blocks(full_response.strip())
    return {
        "choices": [{
            "message": {"role": "assistant", "content": response_text}
        }]
    }
    
@app.post("/api/chat")
async def queue_agent_task(request: Request):
    payload = await request.json()
    response_future = asyncio.Future()
    await task_queue.put(ChatTask(payload, response_future))
    return await response_future

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)

# class AgentInput(BaseModel):
#     prompt: str

# import json

# task_queue = asyncio.Queue()
# NUM_WORKERS = 2

# # Statistiques basiques
# queue_counter = 0

# import termcolor
# from termcolor import colored

# def print_message(msg):
#     msg_type = type(msg).__name__
#     prefix = {
#         "SystemMessage": colored("[SYS]", "cyan"),
#         "HumanMessage": colored("[USER]", "green"),
#         "AIMessage": colored("[AGENT]", "yellow"),
#     }.get(msg_type, "[MSG]")
#     print(f"{prefix} {msg.content.strip()}")

# @app.post("/api/chat")
# async def run_agent_ollama_format(request: Request):

#     payload = await request.json()
#     messages = payload.get("messages", [])

#     formatted_messages: list[BaseMessage] = [
#         ROLE_MAP[m["role"]](content=m["content"])
#         for m in messages
#         if m["role"] in ROLE_MAP
#     ]

#     if not any(isinstance(m, SystemMessage) for m in formatted_messages):
#         formatted_messages.insert(0, SystemMessage(content=prompt.system_prompt))
#     config = {"configurable": {"thread_id": "default"}}

#     try:
#         full_response = ""
#         for i, step in enumerate(agent.stream({"messages": formatted_messages}, config=config, stream_mode="values")):
#             for msg in step["messages"]:
#                 print_message(msg)
#             if isinstance(msg, AIMessage):
#                 full_response += msg.content + "\n"

#         print("#### Final Response ####")
#         response_text = prompt.remove_multiline_think_blocks(full_response.strip())
#         print(f"{response_text}\n")
#         return {
#             "choices": [
#                 {
#                     "message": {
#                         "role": "assistant",
#                         "content": response_text
#                     }
#                 }
#             ]
#         }
#     except ValueError as e:
#         return {
#             "choices": [
#                 {
#                     "message": {
#                     "role": "assistant",
#                     "content": f"Error: {str(e)}"
#                     }
#                 }
#             ]
#         }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)