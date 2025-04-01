from limin import (
    generate_text_completion,
    Conversation,
    Message,
    generate_text_completion_for_conversation,
)


async def main():
    completion = await generate_text_completion(
        "Erkläre mit den Unterschied zwischen 'dass', 'das' und 'daß'.",
        log_probs=True,
        top_log_probs=10,
    )
    print(completion.message)
    print(completion.token_log_probs)
    print(completion.to_pretty_log_probs_string(show_probabilities=True))
    print(completion.to_pretty_log_probs_string(show_probabilities=False))


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
