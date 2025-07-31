import math
from pydantic import BaseModel
from openai.types.chat.chat_completion import Choice


class TokenLogProb(BaseModel):
    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return math.exp(self.log_prob)

    def __repr__(self) -> str:
        return f"TokenLogProb(token={self.token!r}, prob={round(self.prob, 2)})"


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
