from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model(
    model="qwen3:1.7b",
    model_provider="ollama",
    temperature=0.1,
    max_tokens=256)

# memory = MemorySaver()
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools)

system_template = "You are an autonomous agent that can search the web to answer questions. You need to respect the user queries. If he wants you to respond in a certain format you will do it."

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

config = {"configurable": {"thread_id": "abc123"}}

def run_agent():
    # prompt = "Hey I live in Paris, I want you to tell me all the places I can visit in Paris."
    # system_msg = SystemMessage(content=system_template)
    # user_msg = HumanMessage(content=prompt)
    # messages = [system_msg, user_msg]
    # prompt = prompt_template.invoke({"text": prompt})
    # messages = prompt.to_messages()

    for step in agent_executor.stream(
        {"messages": prompt},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


run_agent()
