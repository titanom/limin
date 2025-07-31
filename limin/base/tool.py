from typing import Callable
import typing
from pydantic import BaseModel
from openai.types.chat import ChatCompletionToolParam


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: type[BaseModel],
        exec_fn: Callable[..., str] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.exec_fn = exec_fn

    def execute(self, **kwargs) -> str:
        if self.exec_fn is None:
            raise ValueError("Tool has no execution function")
        return self.exec_fn(**kwargs)

    @property
    def openai_tool(self) -> ChatCompletionToolParam:
        model_json_schema = self.parameters.model_json_schema()
        model_json_schema["additionalProperties"] = False

        return typing.cast(
            ChatCompletionToolParam,
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": model_json_schema,
                },
                "strict": True,
            },
        )
