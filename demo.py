from pydantic import BaseModel
from limin import generate_structured_completion, generate_text_completion


class CapitalModel(BaseModel):
    capital: str


async def main():
    completion = await generate_structured_completion(
        "What is the capital of France?",
        response_model=CapitalModel,
        temperature=0.0,
        log_probs=True,
        top_log_probs=10,
    )
    print(completion.content)
    print(completion.content.capital)
    print(completion.full_token_log_probs)

if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
