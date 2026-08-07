from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        ...

    @abstractmethod
    def get_history(self) -> list[dict]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class InMemoryHistory(Memory):
    def __init__(self):
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
