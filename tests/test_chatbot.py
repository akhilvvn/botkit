from unittest.mock import MagicMock, patch
import pytest


class TestInMemoryHistory:
    def setup_method(self):
        from memory.memory import InMemoryHistory

        self.memory = InMemoryHistory()

    def test_add_and_get(self):
        self.memory.add_message("user", "hello")
        self.memory.add_message("assistant", "hi there")
        history = self.memory.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello"}
        assert history[1] == {"role": "assistant", "content": "hi there"}

    def test_clear(self):
        self.memory.add_message("user", "hello")
        self.memory.clear()
        assert self.memory.get_history() == []

    def test_returns_copy(self):
        self.memory.add_message("user", "hello")
        history = self.memory.get_history()
        history.append({"role": "user", "content": "extra"})
        assert len(self.memory.get_history()) == 1


class TestToolRegistry:
    def setup_method(self):
        from tools.registry import ToolRegistry, current_time

        self.registry = ToolRegistry()
        self.current_time = current_time

    def test_register_and_get(self):
        self.registry.register(self.current_time)
        tools = self.registry.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "current_time"

    def test_execute(self):
        self.registry.register(self.current_time)
        result = self.registry.execute("current_time", {})
        assert isinstance(result, str)

    def test_execute_unknown_tool(self):
        with pytest.raises(ValueError, match="Tool not found"):
            self.registry.execute("nonexistent", {})

    def test_returns_copy(self):
        self.registry.register(self.current_time)
        tools = self.registry.get_tools()
        tools.clear()
        assert len(self.registry.get_tools()) == 1


class TestLLMProviders:
    @patch("langchain_ollama.ChatOllama")
    def test_ollama_provider(self, mock_cls):
        from llm.provider import OllamaProvider

        provider = OllamaProvider(model="test-model")
        provider.get_chat_model()
        mock_cls.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:11434",
            temperature=0.7,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_lmstudio_provider(self, mock_cls):
        from llm.provider import LMStudioProvider

        provider = LMStudioProvider(model="test-model")
        provider.get_chat_model()
        mock_cls.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:1234/v1",
            temperature=0.7,
            api_key="not-needed",
        )


class TestRetriever:
    def test_add_and_query(self):
        from retrieval.retriever import LangChainRetriever

        retriever = LangChainRetriever()
        retriever.add_documents([
            "Python is a programming language.",
            "The sun is a star in our solar system.",
            "LangChain is a framework for LLM applications.",
        ])
        results = retriever.query("programming language", top_k=1)
        assert len(results) >= 1
        assert "Python" in results[0]


class TestChatbotFactory:
    def test_invalid_provider(self):
        from core.chatbot import create_chatbot

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_chatbot({"llm_provider": "nonexistent"})

    @patch("llm.provider.OllamaProvider.get_chat_model")
    def test_create_minimal(self, mock_get_model):
        mock_get_model.return_value = MagicMock()
        from core.chatbot import create_chatbot

        bot = create_chatbot({
            "llm_provider": "ollama",
            "enable_retrieval": False,
            "enable_tools": False,
        })
        assert bot.retriever is None
        assert bot.tool_registry is None

    @patch("llm.provider.OllamaProvider.get_chat_model")
    def test_chat_returns_response(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        from core.chatbot import create_chatbot

        bot = create_chatbot({
            "llm_provider": "ollama",
            "enable_retrieval": False,
            "enable_tools": False,
        })
        assert bot.chat("Hi") == "Hello!"

    @patch("llm.provider.OllamaProvider.get_chat_model")
    def test_memory_accumulates(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_response.tool_calls = []
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        from core.chatbot import create_chatbot

        bot = create_chatbot({
            "llm_provider": "ollama",
            "enable_retrieval": False,
            "enable_tools": False,
        })
        bot.chat("first")
        bot.chat("second")
        history = bot.memory.get_history()
        assert len(history) == 4
        assert history[0] == {"role": "user", "content": "first"}
        assert history[2] == {"role": "user", "content": "second"}
