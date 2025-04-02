import asyncio
import math
import time
from typing import Generic, Literal, Type, TypeVar, cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from tqdm import tqdm
from pydantic import BaseModel, Field

from .base import (
    T,
    Conversation,
    StructuredCompletion,
    TextCompletion,
    get_first_element,
    parse_logprobs,
)


async def generate_structured_completion_for_conversation(
    conversation: Conversation,
    response_model: Type[T],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> StructuredCompletion[T]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    start_time = time.time()
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=conversation.openai_messages,
        response_format=response_model,
        temperature=temperature,
        logprobs=log_probs,
        top_logprobs=top_log_probs,
    )
    end_time = time.time()

    first_choice = get_first_element(completion.choices)
    if first_choice is None:
        raise ValueError("No choices returned from the completion.")

    message_content = first_choice.message.parsed

    if message_content is None:
        raise ValueError("No message content returned from the completion.")

    full_token_log_probs = parse_logprobs(first_choice)

    return StructuredCompletion(
        conversation=conversation,
        model=model,
        content=message_content,
        start_time=start_time,
        end_time=end_time,
        full_token_log_probs=full_token_log_probs,
    )


async def generate_structured_completion(
    user_prompt: str,
    response_model: Type[T],
    *,
    model: str = "gpt-4o",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    log_probs: bool = False,
    top_log_probs: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> StructuredCompletion[T]:
    conversation = Conversation.from_prompts(user_prompt, system_prompt)

    return await generate_structured_completion_for_conversation(
        conversation,
        response_model=response_model,
        model=model,
        temperature=temperature,
        log_probs=log_probs,
        top_log_probs=top_log_probs,
        api_key=api_key,
        base_url=base_url,
    )
