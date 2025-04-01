# limin

A Python library for interacting with OpenAI-compatible LLM APIs.

Features:

✅ Parallel generation of text completions.

✅ Timestamping of text completions and measuring generation duration.

✅ Pretty printing of conversations.

✅ Improved type safety and type inference.

✅ Working with log probabilities of tokens (including pretty printing).

## Installation

Install the library using pip:

```bash
python -m pip install limin
```

## Usage

### General Usage Notes

Note that the entire library is asynchronous.
If you want to use it in a script, you can use `asyncio.run` to run the main function.

Additionally, you will need to set the `OPENAI_API_KEY` environment variable to your API key (or pass the `api_key` parameter to the function you want to use).

For example, you can retrieve a text completion for a single user prompt by calling the `generate_text_completion` function:

```python
from limin import generate_text_completion

completion = await generate_text_completion("What is the capital of France?")
print(completion.message)
```

If you want to use this in a script, you can do the following:

```python
from limin import generate_text_completion


async def main():
    completion = await generate_text_completion("What is the capital of France?")
    print(completion.message)


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()

    asyncio.run(main())
```

This will print something like:

```
The capital of France is Paris.
```

### Generating a Single Text Completion

You can generate a single text completion for a user prompt by calling the `generate_text_completion` function:

```python
from limin import generate_text_completion

completion = await generate_text_completion("What is the capital of France?")
print(completion.message)
```

You can generate a single text completion for a conversation by calling the `generate_text_completion_for_conversation` function:

```python
from limin import generate_text_completion_for_conversation

conversation = Conversation(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="The capital of France is Paris."),
        Message(role="user", content="What is the capital of Germany?"),
    ]
)
completion = await generate_text_completion_for_conversation(conversation)
print(completion.message)
```

### Generating Multiple Text Completions

You can generate multiple text completions for a list of user prompts by calling the `generate_text_completions` function:

```python
from limin import generate_text_completions

completions = await generate_text_completions([
    "What is the capital of France?",
    "What is the capital of Germany?",
])

for completion in completions:
    print(completion.message)
```

It's important to note that the `generate_text_completions` function will parallelize the generation of the text completions.
The number of parallel completions is controlled by the `n_parallel` parameter (which defaults to 5).

For example, if you want to generate 4 text completions with 2 parallel completions, you can do the following:

```python
completions = await generate_text_completions([
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
], n_parallel=2)

for completion in completions:
    print(completion.message)
```

You can also generate multiple text completions for a list of conversations by calling the `generate_text_completions_for_conversations` function:

```python
from limin import generate_text_completions_for_conversations

first_conversation = Conversation(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?"),
    ]
)

second_conversation = Conversation(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of Germany?"),
    ]
)

completions = await generate_text_completions_for_conversations([
    first_conversation,
    second_conversation,
], n_parallel=2)

for completion in completions:
    print(completion.message)
```

Note that both the `generate_text_completions` and `generate_text_completions_for_conversations` functions will show a progress bar if the `show_progress` parameter is set to `True` (which it is by default).
You can suppress this by setting the `show_progress` parameter to `False`.

### Extracting Log Probabilities

You can extract the log probabilities of the tokens by accessing the `token_log_probs` attribute of the `TextCompletion` object.
You will need to pass the `log_probs` parameter to the generation function together with the `top_log_probs` parameter to get the most likely tokens:

```python
completion = await generate_text_completion(
    "What is 2+2?",
    log_probs=True,
    top_log_probs=10,
)
print(completion.token_log_probs)
```

This will return a list of `TokenLogProb` objects, which have the following attributes:

- `token`: The token.
- `log_prob`: The log probability of the token.

You can pretty print the log probabilities by calling the `to_pretty_log_probs_string` method of the `TextCompletion` object:

```python
print(completion.to_pretty_log_probs_string(show_probabilities=True))
```

This will return a nicely colored string with the log probabilities of the tokens.

You can also access the full list of log probabilities by accessing the `full_token_log_probs` attribute of the `TextCompletion` object:

```python
print(completion.full_token_log_probs)
```

This will return a list of lists of `TokenLogProb` objects (for each token position the `top_log_probs` number of most likely tokens).

## Important Classes

### The Message Class

The `Message` class is a simple dataclass that represents a message in a conversation.
It has the following attributes:

- `role`: The role of the message (either "system", "user", or "assistant").
- `content`: The content of the message.

### The Conversation Class

The `Conversation` class represents a conversation between a user and an assistant.
It contains the `messages` attribute, which is a list of `Message` objects.

You can add a message to the conversation using the `add_message` method.
This will intelligently check whether the message has the correct role and then add the message to the conversation.

Additionally, the `Conversation` class has a `to_pretty_string` method that returns a pretty string representation of the conversation with colored roles and separators.

### The TextCompletion Class

The generation functions return either a `TextCompletion` object or a list of `TextCompletion` objects.
This has the following attributes:

- `conversation`: The conversation that was used to generate the completion.
- `model`: The model that was used to generate the completion.
- `message`: The message that was generated.
- `start_time`: The start time of the generation.
- `end_time`: The end time of the generation.
- `duration`: The duration of the generation took (in seconds).

The `start_time`, `end_time`, and `duration` attributes allow you to benchmark the performance of the generation.
