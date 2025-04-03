from limin import generate_text_completions


async def main():
    completions = await generate_text_completions(
        [
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?",
            "What is the capital of Spain?",
        ],
        n_parallel=2,
    )

    for completion in completions:
        print(completion.content)


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())
