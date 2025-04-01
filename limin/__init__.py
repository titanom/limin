import asyncio
from dataclasses import dataclass
import math
import time
from typing import Literal, TypeVar, cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm


T = TypeVar("T")


def get_last_element(list: list[T]) -> T | None:
    if len(list) == 0:
        return None
    return list[-1]


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    @property
    def openai_message(self) -> ChatCompletionMessageParam:
        return cast(
            ChatCompletionMessageParam, {"role": self.role, "content": self.content}
        )


@dataclass
class TokenLogProb:
    token: str
    log_prob: float

    @property
    def prob(self) -> float:
        return math.exp(self.log_prob)

    def __repr__(self) -> str:
        return f"TokenLogProb(token={self.token!r}, prob={round(self.prob, 2)})"


class Conversation:
    def __init__(self, messages: list[Message] | None = None):
        self.messages = messages or []

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

        return "\n".join(pretty_lines)

    @property
    def openai_messages(self) -> list[ChatCompletionMessageParam]:
        return [message.openai_message for message in self.messages]


@dataclass
class TextCompletion:
    conversation: Conversation
    model: str
    message: str
    start_time: float
    end_time: float

    """
    A list containing the most likely tokens and their log probabilities for each token position in the message.
    """
    token_log_probs: list[list[TokenLogProb]] | None = None

    @property
    def duration(self) -> float:
        """The duration of the generation in seconds."""
        return self.end_time - self.start_time


async def generate_text_completion_for_conversation(
    conversation: Conversation,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a conversation.

    :param conversation: The conversation to generate a completion for.
    :param model: The model to use for the completion.
    :param temperature: The temperature to use for the completion.
    :param log_probs: Whether to log the probabilities of the tokens.
    :param top_log_probs: The number of top log probabilities to return.
    :param api_key: The API key to use for the completion.
    :param base_url: The base URL to use for the completion.
    :return: A TextCompletion object.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    start_time = time.time()
    completion = await client.chat.completions.create(
        model=model,
        messages=conversation.openai_messages,
        temperature=temperature,
        logprobs=log_probs,
        top_logprobs=top_log_probs,
    )
    end_time = time.time()

    first_choice = completion.choices[0]

    message_content = first_choice.message.content

    if message_content is None:
        raise ValueError("No message content returned from the completion.")

    token_log_probs = None
    if first_choice.logprobs is not None:
        token_log_probs = [
            [
                TokenLogProb(
                    token=token_log_prob.token, log_prob=token_log_prob.logprob
                )
                for token_log_prob in log_probs_content.top_logprobs
            ]
            for log_probs_content in first_choice.logprobs.content
        ]

    return TextCompletion(
        conversation=conversation,
        model=model,
        message=message_content,
        start_time=start_time,
        end_time=end_time,
        token_log_probs=token_log_probs,
    )


async def generate_text_completion(
    user_prompt: str,
    *,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a user prompt.

    :param user_prompt: The user prompt to generate a completion for.
    :param model: The model to use for the completion.
    :param system_prompt: The system prompt to use for the completion.
    :param temperature: The temperature to use for the completion.
    :param log_probs: Whether to log the probabilities of the tokens.
    :param top_log_probs: The number of top log probabilities to return.
    :param api_key: The API key to use for the completion.
    :param base_url: The base URL to use for the completion.
    :return: A TextCompletion object.
    """
    conversation = Conversation()
    if system_prompt:
        conversation.add_message(Message(role="system", content=system_prompt))
    conversation.add_message(Message(role="user", content=user_prompt))

    return await generate_text_completion_for_conversation(
        conversation,
        model=model,
        temperature=temperature,
        log_probs=log_probs,
        top_log_probs=top_log_probs,
        api_key=api_key,
        base_url=base_url,
    )


async def generate_text_completions_for_conversations(
    conversations: list[Conversation],
    n_parallel: int = 5,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    show_progress: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[TextCompletion]:
    """
    Generate a list of text completions for a list of conversations with support for parallel generation.

    :param conversations: The list of conversations to generate completions for.
    :param n_parallel: The number of completions to generate in parallel.
    :param model: The model to use for the completions.
    :param temperature: The temperature to use for the completions.
    :param log_probs: Whether to log the probabilities of the tokens.
    :param top_log_probs: The number of top log probabilities to return.
    :param show_progress: Whether to show a progress bar.
    :param api_key: The API key to use for the completions.
    :param base_url: The base URL to use for the completions.
    :return: A list of TextCompletion objects.
    """
    completions = []

    if show_progress:
        progress_bar = tqdm(total=len(conversations))

    for i in range(0, len(conversations), n_parallel):
        conversations_batch = conversations[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_text_completion_for_conversation(
                    conversation,
                    model=model,
                    temperature=temperature,
                    log_probs=log_probs,
                    top_log_probs=top_log_probs,
                    api_key=api_key,
                    base_url=base_url,
                )
            )
            for conversation in conversations_batch
        ]

        completions_batch = await asyncio.gather(*tasks)
        completions.extend(completions_batch)

        if show_progress:
            progress_bar.update(len(completions_batch))

    if show_progress:
        progress_bar.close()

    return completions


async def generate_text_completions(
    user_prompts: list[str],
    n_parallel: int = 5,
    *,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    show_progress: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[TextCompletion]:
    """
    Generate a list of text completions for a list of user prompts with support for parallel generation.

    :param user_prompts: The list of user prompts to generate completions for.
    :param n_parallel: The number of completions to generate in parallel.
    :param model: The model to use for the completions.
    :param system_prompt: The system prompt to use for the completions.
    :param temperature: The temperature to use for the completions.
    :param log_probs: Whether to log the probabilities of the tokens.
    :param top_log_probs: The number of top log probabilities to return.
    :param show_progress: Whether to show a progress bar.
    :param api_key: The API key to use for the completions.
    :param base_url: The base URL to use for the completions.
    :return: A list of TextCompletion objects.
    """
    conversations: list[Conversation] = []

    for user_prompt in user_prompts:
        conversation = Conversation()
        if system_prompt:
            conversation.add_message(Message(role="system", content=system_prompt))
        conversation.add_message(Message(role="user", content=user_prompt))
        conversations.append(conversation)

    return await generate_text_completions_for_conversations(
        conversations,
        n_parallel=n_parallel,
        model=model,
        temperature=temperature,
        log_probs=log_probs,
        top_log_probs=top_log_probs,
        show_progress=show_progress,
        api_key=api_key,
        base_url=base_url,
    )
