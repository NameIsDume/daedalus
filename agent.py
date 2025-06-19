from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from model import model
# from tools import tools_list

app = FastAPI()

# def my_pre_model_hook(state):
#     last_msg = state["messages"][-1].content.lower()
#     if "paris" in last_msg:
#         raise ValueError("Query blocked: forbidden keyword 'paris'")
#     return state

agent = create_react_agent(
    model=model,
    # tools=tools_list,
    tools=[],
    checkpointer=MemorySaver(),
    # pre_model_hook=my_pre_model_hook,
)

system_prompt = (
    "You are an autonomous agent"
    "You need to respect the user queries. If they want you to respond in a certain format, you will do it."
)

class AgentInput(BaseModel):
    prompt: str

@app.post("/api/chat")
async def run_agent_ollama_format(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    model_name = payload.get("model", "unknown")
    stream = payload.get("stream", False)

    print(f"Received prompt: {messages}")
    print(f"Model: {model_name}, Stream: {stream}")

    # Formate les messages pour langgraph
    formatted_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            formatted_messages.append(SystemMessage(content=content))
        elif role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))

    config = {"configurable": {"thread_id": "default"}}
    try:
        full_response = ""
        for step in agent.stream({"messages": formatted_messages}, config=config, stream_mode="values"):
            last_msg = step["messages"][-1]
            if isinstance(last_msg, AIMessage):
                full_response += last_msg.content + "\n"

        print(f"Response: {full_response.strip()}\n")
        return {
            "message": {
                "role": "assistant",
                "content": full_response.strip()
            }
        }

    except ValueError as e:
        return {
            "message": {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)