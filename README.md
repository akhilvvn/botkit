# BotKit

A reusable Python chatbot template built on **LangChain**. Clone this repo, edit a few files, and you have a production-ready chatbot with RAG, tools, and conversation memory — no plumbing from scratch.

> **This is a template.** Clone it, rename it, and customize.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally **or** [LM Studio](https://lmstudio.ai) with a loaded model
- Pull a model (if using Ollama):
  ```bash
  ollama pull gemma4:12b
  ```

### Install & Run

```bash
git clone <this-repo> my-chatbot
cd my-chatbot
pip install -e ".[dev]"
python example_app.py
```

The example app starts an interactive terminal chatbot with RAG and tool support.

---

## Architecture

```
example_app.py
    │
    ▼
create_chatbot(config)          ← Factory function (src/core/chatbot.py)
    │
    ├── LLMProvider             ← Abstract interface (src/llm/provider.py)
    │   ├── OllamaProvider      ← Default: local Ollama
    │   └── LMStudioProvider    ← Alternative: LM Studio (OpenAI-compatible)
    │
    ├── Memory                  ← Abstract interface (src/memory/memory.py)
    │   └── InMemoryHistory     ← Default: in-memory list
    │
    ├── Retriever               ← Abstract interface (src/retrieval/retriever.py)
    │   └── LangChainRetriever  ← Default: InMemoryVectorStore + sentence-transformers
    │
    ├── ToolRegistry            ← Tool management (src/tools/registry.py)
    │   └── current_time        ← Example tool
    │
    └── System Prompt           ← Parameterized template (src/prompts/system_prompt.txt)
```

Every layer is behind an abstract interface. Swap any piece by writing a new class that inherits the ABC — nothing else needs to change.

---

## Config Reference

The factory function `create_chatbot()` accepts a plain dict:

```python
config = {
    # LLM
    "llm_provider": "ollama",              # "ollama" or "lmstudio"
    "model": "gemma4:12b",                 # Model name for the chosen provider
    "base_url": "http://localhost:11434",   # Provider endpoint
    "temperature": 0.7,

    # Persona
    "bot_name": "BotKit Assistant",
    "bot_personality": "helpful, concise, and friendly",

    # RAG (opt-in)
    "enable_retrieval": True,
    "embedding_model": "all-MiniLM-L6-v2",

    # Tools (opt-in)
    "enable_tools": True,

    # Prompt
    "system_prompt_path": "src/prompts/system_prompt.txt",  # Optional override
}
```

---

## Adapting This Template

### Rename the project

1. Update `name` in `pyproject.toml`
2. Update `bot_name` in your config

### Swap the LLM provider

Create a new class in `src/llm/provider.py`:

```python
class MyCloudProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    def get_chat_model(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, api_key=self.api_key)
```

Then add an `elif` branch in `create_chatbot()` in `src/core/chatbot.py`.

### Add a new retriever (e.g. LlamaIndex)

Create a new class in `src/retrieval/retriever.py`:

```python
class LlamaIndexRetriever(Retriever):
    def __init__(self, index_path: str):
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        documents = SimpleDirectoryReader(index_path).load_data()
        self._index = VectorStoreIndex.from_documents(documents)
        self._query_engine = self._index.as_query_engine()

    def add_documents(self, texts: list[str]) -> None:
        # Re-index or append to the existing index
        ...

    def query(self, text: str, top_k: int = 3) -> list[str]:
        response = self._query_engine.query(text)
        return [str(response)]
```

No changes needed in `chatbot.py` — just instantiate the new retriever in the factory.

### Add a new tool

Define it with `@tool` in `src/tools/registry.py`:

```python
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # your implementation
    return results
```

Then register it in the factory's `enable_tools` block in `src/core/chatbot.py`:

```python
tool_registry.register(web_search)
```

### Change the memory backend

Create a new class in `src/memory/memory.py`:

```python
class RedisHistory(Memory):
    def __init__(self, redis_url: str, session_id: str):
        import redis
        self._client = redis.from_url(redis_url)
        self._key = f"chat:{session_id}"

    def add_message(self, role: str, content: str) -> None:
        self._client.rpush(self._key, json.dumps({"role": role, "content": content}))

    def get_history(self) -> list[dict]:
        return [json.loads(m) for m in self._client.lrange(self._key, 0, -1)]

    def clear(self) -> None:
        self._client.delete(self._key)
```

### Customize the system prompt

Edit `src/prompts/system_prompt.txt`. Available placeholders:

| Placeholder | Filled at | Source |
|---|---|---|
| `{bot_name}` | Startup | `config["bot_name"]` |
| `{bot_personality}` | Startup | `config["bot_personality"]` |
| `{context}` | Each turn | RAG retrieval results |

---

## What NOT to edit

| File | Why |
|---|---|
| `src/core/chatbot.py` | Only edit the factory function to add new provider/retriever branches |
| `src/memory/memory.py` (ABC) | The interface is stable — add new classes, don't change the ABC |
| `src/retrieval/retriever.py` (ABC) | Same — add implementations, don't change the contract |
| `src/llm/provider.py` (ABC) | Same |
| `tests/conftest.py` | Path setup — leave as-is |

---

## Running Tests

```bash
pytest -v
```

All tests mock LLM calls, so they run without a running model. The retriever test downloads a small embedding model (~23 MB) on first run.

---

## Project Structure

```
botkit/
├── pyproject.toml              ← Dependencies
├── example_app.py              ← Runnable demo
├── README.md
├── src/
│   ├── core/
│   │   └── chatbot.py          ← Factory + Chatbot class
│   ├── llm/
│   │   └── provider.py         ← LLM provider interface + Ollama/LMStudio
│   ├── retrieval/
│   │   └── retriever.py        ← Retriever interface + LangChain implementation
│   ├── memory/
│   │   └── memory.py           ← Memory interface + in-memory implementation
│   ├── tools/
│   │   └── registry.py         ← Tool registry + example tool
│   └── prompts/
│       └── system_prompt.txt   ← Parameterized system prompt
└── tests/
    ├── conftest.py
    └── test_chatbot.py
```
