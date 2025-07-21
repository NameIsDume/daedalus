# from fastapi import FastAPI
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver
# from pydantic import BaseModel
# from typing import Literal, Dict
# import httpx
# from typing import TypedDict, List
# from langchain_core.messages import BaseMessage
# from termcolor import colored
# import uvicorn
# from model import model_llm
# from tools import linux_doc_node, search_in_doc_node

# # ==========================================
# # 1. Définir les Schemas Pydantic
# # ==========================================
# class PlanOutput(BaseModel):
#     action: Literal["linux_doc", "search_in_doc", "finish"]
#     input: Dict[str, str]

# class AgentState(TypedDict):
#     messages: List[BaseMessage]
#     plan: PlanOutput

# class ChatInput(BaseModel):
#     prompt: str
#     thread_id: str = "default"

# # ==========================================
# # 3. Initialiser le modèle
# # ==========================================
# llm = model_llm  # Remplace par ton modèle local (ex: Ollama)

# import re

# def extract_json(text: str) -> str:
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if match:
#         return match.group(0)
#     raise ValueError("No JSON found in LLM output")

# # ==========================================
# # 4. LLM Planner Node
# # ==========================================
# def llm_planner(state: Dict) -> Dict:
#     """Planifie le prochain outil ou termine."""
#     messages = state["messages"]
#     prompt = """
# You are an autonomous Linux assistant. You have two tools:
# - linux_doc(command): Fetch the manual page of a Linux command.
# - search_in_doc(command, keyword): Search for a specific keyword inside the manual page of a command.

# Important Rules:
# - First, check if the user question is related to Linux commands or terminal usage.
# - If the question is NOT about Linux or shell commands, answer immediately and do NOT call any tool.
# - If the question IS about Linux commands:
#     1. You MUST call linux_doc first for the most relevant command.
#     2. After calling linux_doc, if needed, use search_in_doc ONLY if the question asks about a specific option or flag, or if you need to locate a specific keyword in the manual.
#     3. NEVER invent keywords that do not appear in the user question.
#     4. If the answer is clear after linux_doc, skip search_in_doc and go to finish.
# - NEVER return finish as your first action in a Linux-related question.

# Output ONLY JSON, with no additional text:
# {"action": "linux_doc"|"search_in_doc"|"finish", "input": {...}}

# Examples:
# {"action": "linux_doc", "input": {"command": "ls"}}
# {"action": "search_in_doc", "input": {"command": "ls", "keyword": "hidden"}}
# {"action": "finish", "input": {"answer": "Use ls -a to display hidden files."}}
# """
#     response = llm.invoke([SystemMessage(content=prompt)] + messages)

#     # ✅ Extraire uniquement le JSON
#     json_text = extract_json(response.content)
#     parsed = PlanOutput.model_validate_json(json_text)

#     print(colored(f"[PLANNER DECISION] {parsed.action} → input: {parsed.input}", "green", attrs=["bold"]))
#     return {"plan": parsed}

# # ==========================================
# # 5. Node Final Answer
# # ==========================================
# def final_answer(state: Dict) -> Dict:
#     """Construit la réponse finale."""
#     answer = state["plan"].input.get("answer", "No final answer provided.")
#     return {"messages": [HumanMessage(content=f"Final Answer: {answer}")]}

# # ==========================================
# # 6. Construire le Graph LangGraph
# # ==========================================
# graph = StateGraph(AgentState)

# graph.add_node("planner", llm_planner)
# graph.add_node("linux_doc", linux_doc_node)
# graph.add_node("search_in_doc", search_in_doc_node)
# graph.add_node("final_answer", final_answer)

# # Edges
# def route_from_plan(state: AgentState) -> str:
#     """Détermine le prochain nœud en fonction du plan."""
#     if state["plan"].action == "linux_doc":
#         return "linux_doc"
#     elif state["plan"].action == "search_in_doc":
#         return "search_in_doc"
#     elif state["plan"].action == "finish":
#         return "final_answer"
#     return "final_answer"  # fallback

# # 2. Ajouter des conditional edges
# graph.add_conditional_edges(
#     "planner",
#     route_from_plan,
#     {
#         "linux_doc": "linux_doc",
#         "search_in_doc": "search_in_doc",
#         "final_answer": "final_answer"
#     }
# )

# graph.add_edge("linux_doc", "planner")
# graph.add_edge("search_in_doc", "planner")
# graph.add_edge("final_answer", END)
# graph.set_entry_point("planner")

# # Compile
# checkpointer = MemorySaver()
# app_graph = graph.compile(checkpointer=checkpointer)

# # ==========================================
# # 7. FastAPI Server
# # ==========================================
# app = FastAPI(title="Linux Agent API")

# # @app.post("/api/chat")
# # async def chat_endpoint(input: ChatInput):
# #     thread_id = getattr(input, "thread_id", "default")
# #     result = app_graph.invoke(
# #         {"messages": [HumanMessage(content=input.prompt)]},
# #         config={"thread_id": thread_id}
# #     )
# #     return {
# #         "choices": [
# #             {"message": {"role": "assistant", "content": result["messages"][-1].content}}
# #         ]
# #     }
# # ==========================================
# # 8. Lancement avec uvicorn
# # ==========================================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=11435, reload=True)
