import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core.chatbot import create_chatbot


CONFIG = {
    "llm_provider": "ollama",
    "model": "gemma4:12b",
    "temperature": 0.7,
    "bot_name": "BotKit Assistant",
    "bot_personality": "helpful, concise, and friendly",
    "enable_retrieval": True,
    "embedding_model": "all-MiniLM-L6-v2",
    "enable_tools": True,
}

SAMPLE_DOCUMENTS = [
    "BotKit is a reusable Python chatbot template built with LangChain.",
    "To add a new LLM provider, create a class implementing LLMProvider in src/llm/provider.py.",
    "The retrieval layer uses an abstract Retriever interface so backends like LlamaIndex can be swapped in.",
    "Tools are registered via ToolRegistry in src/tools/registry.py using LangChain's @tool decorator.",
    "BotKit supports Ollama and LMStudio as local LLM providers.",
]


def main():
    print("Building chatbot...")
    bot = create_chatbot(CONFIG)

    if bot.retriever:
        bot.retriever.add_documents(SAMPLE_DOCUMENTS)
        print("Knowledge base loaded with sample documents.")

    print("\n--- BotKit Assistant ---")
    print("Type 'quit' to exit, 'clear' to reset conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            bot.memory.clear()
            print("Conversation cleared.\n")
            continue

        try:
            response = bot.chat(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
