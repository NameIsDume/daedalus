import asyncio
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from agent_manager import handle_request, get_memory, cleanup_inactive_threads
NUM_WORKERS = 2

task_queue = asyncio.Queue()
active_threads = set()
queue_counter = 0

class ChatTask:
    def __init__(self, payload, response_future):
        self.payload = payload
        self.response_future = response_future
        self.id = uuid.uuid4().hex
        self.thread_id = payload.get("thread_id") or self.id

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔧 Launching {NUM_WORKERS} workers...")
    for i in range(NUM_WORKERS):
        asyncio.create_task(agent_worker(i))
    asyncio.create_task(cleanup_inactive_threads(ttl=10))  # TTL = 10 min
    yield
    print("🛑 Shutting down workers...")

app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def queue_agent_task(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse(content={"error": "Invalid payload"}, status_code=400)
    response_future = asyncio.Future()
    await task_queue.put(ChatTask(payload, response_future))
    return await response_future

@app.get("/api/status")
async def status() -> dict:
    return {"pending": task_queue.qsize()}

@app.get("/api/debug_memory")
async def debug_memory(thread_id: str = "default") -> JSONResponse:
    state = get_memory(thread_id)
    if not state:
        return JSONResponse(content={thread_id: "🕳️ Empty state"})

    messages = state["channel_values"]["messages"]
    content_list = [{"role": msg.__class__.__name__.replace("Message", "").lower(), "content": msg.content} for msg in messages]

    return JSONResponse(content={thread_id: content_list})

async def agent_worker(worker_id) -> None:
    while True:
        task = await task_queue.get()
        try:
            result = await handle_request(task.thread_id, task.payload)
            task.response_future.set_result(result)
        except Exception as e:
            task.response_future.set_result({"error": str(e)})
        task_queue.task_done()