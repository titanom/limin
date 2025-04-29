import math
from typing import Generic, Literal, TypeVar, cast
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, Field

T = TypeVar("T")


def get_first_element(list: list[T]) -> T | None:
    if len(list) == 0:
        return None
    return list[0]


def get_last_element(list: list[T]) -> T | None:
    if len(list) == 0:
        return None
    return list[-1]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return cast(
            ChatCompletionMessageParam, {"role": self.role, "content": self.content}
        )


class TokenLogProb(BaseModel):
    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return math.exp(self.log_prob)

    def __repr__(self) -> str:
        return f"TokenLogProb(token={self.token!r}, prob={round(self.prob, 2)})"


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

        if last_message.role == "assistant" and message.role != "user":
            raise ValueError("Assistant message must be followed by a user message")

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
            conversation.add_message(Message(role="system", content=system_prompt))
        conversation.add_message(Message(role="user", content=user_prompt))
        if assistant_prompt is not None:
            conversation.add_message(
                Message(role="assistant", content=assistant_prompt)
            )
        return conversation


def format_token_log_probs(
    token_log_probs: list[TokenLogProb], show_probabilities: bool = False
) -> str:
    """
    Returns a pretty string representation of the token log probabilities.
    Tokens are colored from dark red (low probability) to dark green (high probability).

    :param token_log_probs: List of TokenLogProb objects to format
    :param show_probabilities: Whether to show the probability value after each token.
    """
    result = []
    for token_log_prob in token_log_probs:
        if token_log_prob.prob < 0.25:
            color_code = "\033[1;31m"  # Dark red
        elif token_log_prob.prob < 0.5:
            color_code = "\033[1;33m"  # Yellow
        elif token_log_prob.prob < 0.75:
            color_code = "\033[1;32m"  # Light green
        else:
            color_code = "\033[1;92m"  # Dark green

        # Reset color code
        reset_code = "\033[0m"

        if show_probabilities:
            result.append(
                f"{color_code}{token_log_prob.token}[{round(token_log_prob.prob, 2)}]{reset_code}"
            )
        else:
            result.append(f"{color_code}{token_log_prob.token}{reset_code}")

    return "".join(result)


class TextCompletion(BaseModel):
    conversation: Conversation
    model: str
    content: str
    start_time: float
    end_time: float

    """
    A list containing the most likely tokens and their log probabilities for each token position in the message.
    """
    full_token_log_probs: list[list[TokenLogProb]] | None = None

    @property
    def duration(self) -> float:
        """The duration of the generation in seconds."""
        return self.end_time - self.start_time

    @property
    def token_log_probs(self) -> list[TokenLogProb] | None:
        if self.full_token_log_probs is None:
            return None

        return [
            token_log_probs_position[0]
            for token_log_probs_position in self.full_token_log_probs
        ]

    def to_pretty_log_probs_string(self, show_probabilities: bool = False) -> str:
        """
        Returns a pretty string representation of the token log probabilities.
        Tokens are colored from dark red (low probability) to dark green (high probability).

        :param show_probabilities: Whether to show the probability value after each token.
        """
        if self.token_log_probs is None:
            return "No token log probabilities available."

        return format_token_log_probs(self.token_log_probs, show_probabilities)


class StructuredCompletion(BaseModel, Generic[T]):
    conversation: Conversation
    model: str
    content: T
    start_time: float
    end_time: float
    full_token_log_probs: list[list[TokenLogProb]] | None = None

    @property
    def duration(self) -> float:
        """The duration of the generation in seconds."""
        return self.end_time - self.start_time

    @property
    def token_log_probs(self) -> list[TokenLogProb] | None:
        if self.full_token_log_probs is None:
            return None

        return [
            token_log_probs_position[0]
            for token_log_probs_position in self.full_token_log_probs
        ]

    def to_pretty_log_probs_string(self, show_probabilities: bool = False) -> str:
        if self.token_log_probs is None:
            return "No token log probabilities available."

        return format_token_log_probs(self.token_log_probs, show_probabilities)


def parse_logprobs(first_choice: Choice) -> list[list[TokenLogProb]] | None:
    token_log_probs = None
    if first_choice.logprobs is not None and first_choice.logprobs.content is not None:
        token_log_probs = [
            [
                TokenLogProb(
                    token=token_log_prob.token, log_prob=token_log_prob.logprob
                )
                for token_log_prob in log_probs_content.top_logprobs
            ]
            for log_probs_content in first_choice.logprobs.content
        ]
    return token_log_probs


class ModelConfiguration(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 1.0
    log_probs: bool = False
    top_log_probs: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_p: float | None = None
    seed: int | None = None
    api_key: str | None = None
    base_url: str | None = None


DEFAULT_MODEL_CONFIGURATION = ModelConfiguration()
