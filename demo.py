from pydantic import BaseModel
from limin import generate_structured_completion, generate_text_completion


class CapitalModel(BaseModel):
    capital: str


async def main():
    completion = await generate_structured_completion(
        "What is the capital of France?",
        response_model=CapitalModel,
    )
    print(completion.content)
    print(completion.content.capital)


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
