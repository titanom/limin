from abc import ABC, abstractmethod
from pydantic import BaseModel
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import Function


class Tool(ABC):
    name: str
    description: str
    parameters: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    @property
    def openai_tool(self) -> ChatCompletionToolParam:
        model_json_schema = self.parameters.model_json_schema()
        model_json_schema["additionalProperties"] = False

        return ChatCompletionToolParam(
            type="function",
            function=Function(
                name=self.name,
                description=self.description,
                parameters=model_json_schema,
                strict=True,
            ),
        )
