from pydantic import BaseModel
from typing import Literal, Optional, Union, cast
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
import json


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return ChatCompletionMessageParam(
            role=self.role,
            content=self.content,
        )


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return ChatCompletionMessageParam(
            role=self.role,
            content=self.content,
        )


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict

    @property
    def openai_tool_call(self) -> ChatCompletionMessageToolCall:
        return ChatCompletionMessageToolCall(
            id=self.id,
            function=Function(name=self.name, arguments=json.dumps(self.arguments)),
            type="function",
        )


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str]
    tool_calls: list[ToolCall]

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return ChatCompletionMessageParam(
            role=self.role,
            content=self.content,
            tool_calls=[tool_call.openai_tool_call for tool_call in self.tool_calls],
        )


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return ChatCompletionMessageParam(
            role=self.role, content=self.content, tool_call_id=self.tool_call_id
        )


Message = Union[SystemMessage | UserMessage | AssistantMessage | ToolMessage]
