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

# 📡 Endpoint POST
@app.post("/run")
async def run_agent(data: AgentInput):
    prompt = data.prompt
    config = {"configurable": {"thread_id": "default"}}
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        full_response = ""
        for step in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            last_msg = step["messages"][-1]
            if isinstance(last_msg, AIMessage):
                full_response += last_msg.content + "\n"
        return {"response": full_response.strip()}
    except ValueError as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="127.0.0.1", port=11435, reload=True)