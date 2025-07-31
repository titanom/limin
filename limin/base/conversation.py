from pydantic import BaseModel, Field
from .message import Message, SystemMessage, UserMessage, AssistantMessage
from .base_util import get_last_element
from openai.types.chat import ChatCompletionMessageParam


class Conversation(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    def add_message(self, message: Message):
        last_message = get_last_element(self.messages)

        if last_message is None:
            if message.role == "assistant":
                raise ValueError("The first message must be a system or user message")

            self.messages.append(message)
            return

        if last_message.role == "system" and message.role != "user":
            raise ValueError("System message must be followed by a user message")

        if last_message.role == "assistant" and message.role not in ["user", "tool"]:
            raise ValueError(
                "Assistant message must be followed by a user or tool message"
            )

        if last_message.role == "user" and message.role != "assistant":
            raise ValueError("User message must be followed by an assistant message")

        self.messages.append(message)

    def to_pretty_string(
        self,
        system_color: str = "\033[1;36m",
        user_color: str = "\033[1;32m",
        assistant_color: str = "\033[1;35m",
    ) -> str:
        pretty_lines = []

        for message in self.messages:
            if message.role == "system":
                color_code = system_color
            elif message.role == "user":
                color_code = user_color
            elif message.role == "assistant":
                color_code = assistant_color

            # Reset color code
            reset_code = "\033[0m"

            pretty_lines.append(f"{color_code}{message.role.capitalize()}{reset_code}")

            separator_length = len(message.role) + 2  # +2 for some extra space
            pretty_lines.append("-" * separator_length)

            pretty_lines.append(f"{message.content}\n")

        return "\n".join(pretty_lines).strip()

    def to_markdown(self) -> str:
        markdown_str = ""
        for message in self.messages:
            markdown_str += f"## {message.role.capitalize()} \n"
            markdown_str += f"{message.content}\n\n"
        return markdown_str.strip()

    @property
    def openai_messages(self) -> list[ChatCompletionMessageParam]:
        return [message.openai_message for message in self.messages]

    @staticmethod
    def from_prompts(
        user_prompt: str,
        assistant_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> "Conversation":
        conversation = Conversation()
        if system_prompt is not None:
            conversation.add_message(SystemMessage(content=system_prompt))
        conversation.add_message(UserMessage(content=user_prompt))
        if assistant_prompt is not None:
            conversation.add_message(
                AssistantMessage(content=assistant_prompt, tool_calls=[])
            )
        return conversation
