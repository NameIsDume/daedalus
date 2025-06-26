from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()
model = init_chat_model(model="qwen3:1.7b", model_provider="ollama", temperature=0.1, max_tokens=256)
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live in June?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()