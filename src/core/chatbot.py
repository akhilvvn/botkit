from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from llm.provider import LLMProvider, OllamaProvider, LMStudioProvider
from memory.memory import Memory, InMemoryHistory
from retrieval.retriever import Retriever
from tools.registry import ToolRegistry


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class Chatbot:
    def __init__(
        self,
        llm_provider: LLMProvider,
        memory: Memory,
        retriever: Retriever | None = None,
        tool_registry: ToolRegistry | None = None,
        system_prompt_template: str = "",
    ):
        self.llm = llm_provider.get_chat_model()
        self.memory = memory
        self.retriever = retriever
        self.tool_registry = tool_registry
        self.system_prompt_template = system_prompt_template

        if tool_registry and tool_registry.get_tools():
            try:
                self.llm = self.llm.bind_tools(tool_registry.get_tools())
            except Exception:
                self.tool_registry = None

    def chat(self, user_input: str) -> str:
        context = ""
        if self.retriever:
            results = self.retriever.query(user_input)
            if results:
                context = "\n\n".join(results)

        system_content = self.system_prompt_template.replace(
            "{context}", context or "No additional context available."
        )

        messages: list = [SystemMessage(content=system_content)]
        for msg in self.memory.get_history():
            cls = HumanMessage if msg["role"] == "user" else AIMessage
            messages.append(cls(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))

        response = self.llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls and self.tool_registry:
            messages.append(response)
            for tc in response.tool_calls:
                result = self.tool_registry.execute(tc["name"], tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            response = self.llm.invoke(messages)

        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", response.content)
        return response.content


def create_chatbot(config: dict) -> Chatbot:
    provider_name = config.get("llm_provider", "ollama")

    if provider_name == "ollama":
        llm_provider = OllamaProvider(
            model=config.get("model", "gemma4:12b"),
            base_url=config.get("base_url", "https://ollama-vault.infospica.in"),
            temperature=config.get("temperature", 0.7),
        )
    elif provider_name == "lmstudio":
        llm_provider = LMStudioProvider(
            model=config.get("model", "google/gemma-4-12b-qat"),
            base_url=config.get("base_url", "http://localhost:1234/v1"),
            temperature=config.get("temperature", 0.7),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    memory = InMemoryHistory()

    retriever = None
    if config.get("enable_retrieval", False):
        from retrieval.retriever import LangChainRetriever

        retriever = LangChainRetriever(
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
        )

    tool_registry = None
    if config.get("enable_tools", False):
        from tools.registry import current_time

        tool_registry = ToolRegistry()
        tool_registry.register(current_time)

    prompt_path = config.get("system_prompt_path", _PROMPT_DIR / "system_prompt.txt")
    template = Path(prompt_path).read_text()
    template = template.replace("{bot_name}", config.get("bot_name", "BotKit Assistant"))
    template = template.replace(
        "{bot_personality}", config.get("bot_personality", "helpful, concise, and friendly")
    )

    return Chatbot(
        llm_provider=llm_provider,
        memory=memory,
        retriever=retriever,
        tool_registry=tool_registry,
        system_prompt_template=template,
    )
