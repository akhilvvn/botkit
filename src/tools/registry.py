from datetime import datetime
from langchain_core.tools import tool


class ToolRegistry:
    def __init__(self):
        self._tools: list = []

    def register(self, tool_fn) -> None:
        self._tools.append(tool_fn)

    def get_tools(self) -> list:
        return list(self._tools)

    def execute(self, tool_name: str, tool_args: dict) -> str:
        for t in self._tools:
            if t.name == tool_name:
                return str(t.invoke(tool_args))
        raise ValueError(f"Tool not found: {tool_name}")


@tool
def current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
