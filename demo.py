from limin import Conversation, Message, generate_text_completion, generate_text_completion_for_conversation


async def main():
    message = "Tell me a little story (4 sentences) about a cat."
    completion = await generate_text_completion(message, model="gpt-4o", temperature=0.0000000000000001, log_probs=True, top_log_probs=10)
    print(completion.to_pretty_log_probs_string(show_probabilities=True))


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
