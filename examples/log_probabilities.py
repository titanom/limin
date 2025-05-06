import asyncio
from limin import generate_text_completion, ModelConfiguration


async def main():
    model_configuration = ModelConfiguration(
        model="gpt-4o",
        log_probs=True,
        top_log_probs=10,
    )
    completion = await generate_text_completion(
        "What is 2+2?",
        model_configuration=model_configuration,
    )
    print("Token log probabilities:")
    print(completion.token_log_probs)
    print("Full token log probabilities:")
    print(completion.full_token_log_probs)
    print("Pretty log probabilities (with probabilities):")
    print(completion.to_pretty_log_probs_string(show_probabilities=True))
    print("Pretty log probabilities (without probabilities):")
    print(completion.to_pretty_log_probs_string(show_probabilities=False))


if __name__ == "__main__":
    asyncio.run(main())
