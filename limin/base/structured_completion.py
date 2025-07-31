from typing import Generic, TypeVar
from pydantic import BaseModel
from .logprobs import TokenLogProb, format_token_log_probs
from .message import AssistantMessage

T = TypeVar("T")


class StructuredCompletion(BaseModel, Generic[T]):
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

    def to_assistant_message(self) -> AssistantMessage:
        return AssistantMessage(
            role="assistant",
            content=str(self.content),
            tool_calls=None,
        )
