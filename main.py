from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

model = init_chat_model(model="qwen3:0.6b", model_provider="ollama", temperature=0.1, max_tokens=256)
# llm = ChatOllama(model="qwen3:0.6b", temperature=0.1, max_tokens=256)

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

def main():
    print("Hello from daedalus!")

    # response = model.invoke([
    #     SystemMessage(content="Translate the following from english to Italian:"),
    #     HumanMessage(content="hi!")
    # ])
    # print(response.content)
    
    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
    prompt.to_messages()
    response = model.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    main()
