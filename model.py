from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen3:1.7b",
    # model="qwen2.5-coder:1.5b",
    model_provider="ollama",
    temperature=0.1,
    max_tokens=256)
