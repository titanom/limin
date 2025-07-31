from .logprobs import TokenLogProb, format_token_log_probs
from .message import AssistantMessage
from pydantic import BaseModel


class TextCompletion(BaseModel):
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

    def to_assistant_message(self) -> AssistantMessage:
        return AssistantMessage(
            role="assistant",
            content=self.content,
            tool_calls=None,
        )
