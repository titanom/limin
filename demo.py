from limin import (
    generate_text_completion,
    Conversation,
    Message,
    generate_text_completion_for_conversation,
)


async def main():
    conversation = Conversation(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(
                role="user",
                content="Generate a random number between 1 and 10. Only produce the number, no other text.",
            ),
        ]
    )
    completion = await generate_text_completion_for_conversation(
        conversation, log_probs=True, top_log_probs=10
    )
    print(completion.message)
    print(completion.token_log_probs)


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
