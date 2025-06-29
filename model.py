from langchain.chat_models import init_chat_model

model = init_chat_model(
    # model="qwen3:1.7b",
    # model="qwen2.5-coder:1.5b",
    # model="qwen3:"
    # model="qwen3:0.6b",
    model="qwen3:1.7b",
    # model="qwen3:4b",
    # model="qwen3:8b",
    # model="qwen3:14b",
    # model="qwen3:32b",
    # model="qwen3:30b-a3b"
    model_provider="ollama",
    temperature=0.1,
    max_tokens=256)
