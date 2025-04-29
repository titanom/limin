import asyncio
import time
from openai import AsyncOpenAI
from tqdm import tqdm

from .base import (
    DEFAULT_MODEL_CONFIGURATION,
    Conversation,
    ModelConfiguration,
    TextCompletion,
    get_first_element,
    parse_logprobs,
)


async def generate_text_completion_for_conversation(
    conversation: Conversation,
    model_configuration: ModelConfiguration | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a conversation.

    :param conversation: The conversation to generate a completion for.
    :param model_configuration: The model configuration to use for the completion.
    :return: A TextCompletion object.
    """
    if model_configuration is None:
        model_configuration = DEFAULT_MODEL_CONFIGURATION

    client = AsyncOpenAI(
        api_key=model_configuration.api_key,
        base_url=model_configuration.base_url,
    )

    start_time = time.time()
    completion = await client.chat.completions.create(
        model=model_configuration.model,
        messages=conversation.openai_messages,
        temperature=model_configuration.temperature,
        logprobs=model_configuration.log_probs,
        top_logprobs=model_configuration.top_log_probs,
        max_tokens=model_configuration.max_tokens,
        presence_penalty=model_configuration.presence_penalty,
        frequency_penalty=model_configuration.frequency_penalty,
        top_p=model_configuration.top_p,
        seed=model_configuration.seed,
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
        model=model_configuration.model,
        content=message_content,
        start_time=start_time,
        end_time=end_time,
        full_token_log_probs=full_token_log_probs,
    )


async def generate_text_completion(
    user_prompt: str,
    system_prompt: str | None = None,
    model_configuration: ModelConfiguration | None = None,
) -> TextCompletion:
    """
    Generate a text completion for a user prompt.

    :param user_prompt: The user prompt to generate a completion for.
    :param system_prompt: The system prompt to use for the completion.
    :param model_configuration: The model configuration to use for the completion.
    :return: A TextCompletion object.
    """
    if model_configuration is None:
        model_configuration = DEFAULT_MODEL_CONFIGURATION

    conversation = Conversation.from_prompts(user_prompt, system_prompt=system_prompt)

    return await generate_text_completion_for_conversation(
        conversation,
        model_configuration,
    )


async def generate_text_completions_for_conversations(
    conversations: list[Conversation],
    n_parallel: int = 5,
    model_configuration: ModelConfiguration | None = None,
    show_progress: bool = True,
) -> list[TextCompletion]:
    """
    Generate a list of text completions for a list of conversations with support for parallel generation.

    :param conversations: The list of conversations to generate completions for.
    :param n_parallel: The number of completions to generate in parallel.
    :param model_configuration: The model configuration to use for the completions.
    :param show_progress: Whether to show a progress bar.
    :return: A list of TextCompletion objects.
    """
    completions = []

    progress_bar = None

    if show_progress:
        progress_bar = tqdm(total=len(conversations))

    for i in range(0, len(conversations), n_parallel):
        conversations_batch = conversations[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_text_completion_for_conversation(
                    conversation,
                    model_configuration,
                )
            )
            for conversation in conversations_batch
        ]

        completions_batch = await asyncio.gather(*tasks)
        completions.extend(completions_batch)

        if show_progress and progress_bar is not None:
            progress_bar.update(len(completions_batch))

    if show_progress and progress_bar is not None:
        progress_bar.close()

    return completions


async def generate_text_completions(
    user_prompts: list[str],
    system_prompt: str | None = None,
    n_parallel: int = 5,
    model_configuration: ModelConfiguration | None = None,
    show_progress: bool = True,
) -> list[TextCompletion]:
    """
    Generate a list of text completions for a list of user prompts with support for parallel generation.

    :param user_prompts: The list of user prompts to generate completions for.
    :param system_prompt: The system prompt to use for the completions.
    :param n_parallel: The number of completions to generate in parallel.
    :param model_configuration: The model configuration to use for the completions.
    :param show_progress: Whether to show a progress bar.
    :return: A list of TextCompletion objects.
    """
    conversations = [
        Conversation.from_prompts(user_prompt, system_prompt=system_prompt)
        for user_prompt in user_prompts
    ]

    return await generate_text_completions_for_conversations(
        conversations,
        n_parallel=n_parallel,
        model_configuration=model_configuration,
        show_progress=show_progress,
    )
