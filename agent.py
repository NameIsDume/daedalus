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
from fastapi.responses import JSONResponse

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
task_queue = asyncio.Queue()
NUM_WORKERS = 2
active_threads = set()

queue_counter = 0

memory = MemorySaver()

def post_model_hook(state: dict) -> dict:
    # print("post_model_hook called")
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage):
        last_ai = messages[-1]
        # print("Original AI message:")
        print(last_ai.content)
        cleaned = prompt.remove_multiline_think_blocks(last_ai.content)
        # print("Cleaned AI message:")
        print(cleaned)
        # modification of real message content
        # print("Updating last AI message content")
        last_ai.content = cleaned
    return state

agent = create_react_agent(
    model=model,
    # tools=tools_list,
    tools=[],
    checkpointer=memory,
    post_model_hook=post_model_hook,
    # pre_model_hook=my_pre_model_hook,
)

class ChatTask:
    def __init__(self, payload, response_future):
        self.payload = payload
        self.response_future = response_future
        self.id = uuid.uuid4().hex
        self.thread_id = payload.get("thread_id") or self.id

def print_message(msg):
    pass
    # msg_type = type(msg).__name__
    # prefix = {
    #     "SystemMessage": colored("[SYS]", "cyan"),
    #     "HumanMessage": colored("[USER]", "green"),
    #     "AIMessage": colored("[AGENT]", "yellow"),
    # }.get(msg_type, "[MSG]")
    # print(f"{prefix} {msg.content.strip()}")

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
    thread_id = payload.get("thread_id", "default")
    formatted_messages: list[BaseMessage] = [
        ROLE_MAP[m["role"]](content=m["content"]) for m in messages if m["role"] in ROLE_MAP
    ]

    if not any(isinstance(m, SystemMessage) for m in formatted_messages):
        formatted_messages.insert(0, SystemMessage(content=prompt.system_prompt))

    thread_id = payload.get("thread_id", "default")
    active_threads.add(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    full_response = None
    for step in agent.stream({"messages": formatted_messages}, config=config, stream_mode="values"):
        for msg in step["messages"]:
            print_message(msg)
            if isinstance(msg, AIMessage):
                full_response = msg  # Capture the last AI message
                if "act: finish" in msg.content.lower():
                    print("# Finishing thread:", thread_id)
                    await memory.adelete_thread(thread_id)
                    active_threads.discard(thread_id)

    if full_response is not None:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": full_response.content.strip()
                }
            }]
        }
    else:
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

@app.post("/api/chat")
async def queue_agent_task(request: Request):
    payload = await request.json()
    response_future = asyncio.Future()
    await task_queue.put(ChatTask(payload, response_future))
    return await response_future

@app.get("/api/debug_memory")
async def debug_memory():
    print("Debugging memory state...")
    state = memory.get({"configurable": {"thread_id": "default"}})
    print(state)
    try:
        thread_id = "default"
        state = memory.get({"configurable": {"thread_id": thread_id}})
        if not state:
            return JSONResponse(content={thread_id: "🕳️ Empty state"})

        # Extract messages from the state
        messages = state["channel_values"]["messages"]
        content_list = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = msg.content
            content_list.append({"role": role, "content": content})

        return JSONResponse(content={thread_id: content_list})
    except Exception as e:
        print("❌ Error:", str(e))
        return JSONResponse(content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)
