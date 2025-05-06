import asyncio
from limin import generate_text_completion, ModelConfiguration


async def main():
    model_configuration = ModelConfiguration(
        model="gpt-4o",
        temperature=0.7,
        log_probs=True,
        top_log_probs=10,
    )
    completion = await generate_text_completion(
        "What is the capital of France?", model_configuration=model_configuration
    )
    print(completion.content)


if __name__ == "__main__":
    asyncio.run(main())
