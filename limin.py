import asyncio
from dataclasses import dataclass
import time
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm


@dataclass
class TextCompletion:
    conversation: list[ChatCompletionMessageParam]
    model: str
    message: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """The duration of the generation in seconds."""
        return self.end_time - self.start_time


async def generate_text_completion_for_conversation(
    conversation: list[ChatCompletionMessageParam],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a conversation.

    :param conversation: The conversation (i.e. a list of messages) to generate a completion for.
    :param model: The model to use for the completion.
    :param temperature: The temperature to use for the completion.
    :param api_key: The API key to use for the completion.
    :param base_url: The base URL to use for the completion.
    :return: A TextCompletion object.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    start_time = time.time()
    completion = await client.chat.completions.create(
        model=model, messages=conversation, temperature=temperature
    )
    end_time = time.time()

    message_content = completion.choices[0].message.content
    if message_content is None:
        raise ValueError("No message content returned from the completion.")

    return TextCompletion(
        conversation=conversation,
        model=model,
        message=message_content,
        start_time=start_time,
        end_time=end_time,
    )


async def generate_text_completion(
    user_prompt: str,
    *,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a user prompt.

    :param user_prompt: The user prompt to generate a completion for.
    :param model: The model to use for the completion.
    :param system_prompt: The system prompt to use for the completion.
    :param temperature: The temperature to use for the completion.
    :param api_key: The API key to use for the completion.
    :param base_url: The base URL to use for the completion.
    :return: A TextCompletion object.
    """
    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return await generate_text_completion_for_conversation(
        messages,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


async def generate_text_completions_for_conversations(
    conversations: list[list[ChatCompletionMessageParam]],
    n_parallel: int = 5,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    show_progress: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[TextCompletion]:
    """
    Generate a list of text completions for a list of conversations (where each conversation is a list of messages) with support for parallel generation.

    :param conversations: The list of conversations to generate completions for.
    :param n_parallel: The number of completions to generate in parallel.
    :param model: The model to use for the completions.
    :param temperature: The temperature to use for the completions.
    :param show_progress: Whether to show a progress bar.
    :param api_key: The API key to use for the completions.
    :param base_url: The base URL to use for the completions.
    :return: A list of TextCompletion objects.
    """
    completions = []

    if show_progress:
        progress_bar = tqdm(total=len(conversations))

    for i in range(0, len(conversations), n_parallel):
        messages_batch = conversations[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_text_completion_for_conversation(
                    message_list,
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url,
                )
            )
            for message_list in messages_batch
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
    :param show_progress: Whether to show a progress bar.
    :param api_key: The API key to use for the completions.
    :param base_url: The base URL to use for the completions.
    :return: A list of TextCompletion objects.
    """
    messages: list[list[ChatCompletionMessageParam]] = []

    for user_prompt in user_prompts:
        prompt_messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": user_prompt})
        messages.append(prompt_messages)

    return await generate_text_completions_for_conversations(
        messages,
        n_parallel=n_parallel,
        model=model,
        temperature=temperature,
        show_progress=show_progress,
        api_key=api_key,
        base_url=base_url,
    )
