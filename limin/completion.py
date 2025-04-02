import asyncio
import time
from openai import AsyncOpenAI
from tqdm import tqdm

from .base import (
    Conversation,
    TextCompletion,
    get_first_element,
    parse_logprobs,
)


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

    first_choice = get_first_element(completion.choices)
    if first_choice is None:
        raise ValueError("No choices returned from the completion.")

    message_content = first_choice.message.content

    if message_content is None:
        raise ValueError("No message content returned from the completion.")

    full_token_log_probs = parse_logprobs(first_choice)

    return TextCompletion(
        conversation=conversation,
        model=model,
        content=message_content,
        start_time=start_time,
        end_time=end_time,
        full_token_log_probs=full_token_log_probs,
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
    conversation = Conversation.from_prompts(user_prompt, system_prompt)

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
        conversation = Conversation.from_prompts(user_prompt, system_prompt)
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
