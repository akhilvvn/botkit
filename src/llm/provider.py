from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel


class LLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        ...


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gemma4:12b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def get_chat_model(self) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )


class LMStudioProvider(LLMProvider):
    def __init__(
        self,
        model: str = "google/gemma-4-12b-qat",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def get_chat_model(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            api_key="not-needed",
        )
