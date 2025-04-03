from limin import generate_text_completion


async def main():
    completion = await generate_text_completion("What is the capital of France?")
    print(completion.content)


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
