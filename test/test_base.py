import pytest

from limin import (
    Conversation,
    Message,
    TextCompletion,
    TokenLogProb,
    StructuredCompletion,
)


def test_message_openai_message():
    message = Message(role="user", content="Test message")
    openai_message = message.openai_message

    assert isinstance(openai_message, dict)
    assert openai_message["role"] == "user"
    assert openai_message["content"] == "Test message"

    expected = {"role": "user", "content": "Test message"}
    assert openai_message == expected


def test_token_log_prob_prob():
    token_log_prob = TokenLogProb(token="test", log_prob=0.0)
    assert token_log_prob.prob == 1.0


def test_conversation_add_message():
    conversation = Conversation()

    system_message = Message(role="system", content="System prompt")
    conversation.add_message(system_message)
    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"

    user_message = Message(role="user", content="Hello")
    conversation.add_message(user_message)
    assert len(conversation.messages) == 2
    assert conversation.messages[1].role == "user"

    assistant_message = Message(role="assistant", content="Hi there")
    conversation.add_message(assistant_message)
    assert len(conversation.messages) == 3
    assert conversation.messages[2].role == "assistant"

    user_message2 = Message(role="user", content="How are you?")
    conversation.add_message(user_message2)
    assert len(conversation.messages) == 4
    assert conversation.messages[3].role == "user"


def test_conversation_add_message_invalid_first_message():
    conversation = Conversation()
    assistant_message = Message(role="assistant", content="Invalid first message")
    with pytest.raises(ValueError):
        conversation.add_message(assistant_message)


def test_conversation_add_message_invalid_message_sequence_user_user():
    conversation = Conversation()
    conversation.add_message(Message(role="user", content="Hello"))
    with pytest.raises(ValueError):
        conversation.add_message(Message(role="user", content="Hi there"))


def test_conversation_add_message_invalid_message_sequence_assistant_assistant():
    conversation = Conversation()
    conversation.add_message(Message(role="user", content="Hello"))
    conversation.add_message(Message(role="assistant", content="Hi there"))
    with pytest.raises(ValueError):
        conversation.add_message(Message(role="assistant", content="Hi there"))


def test_conversation_to_pretty_string():
    conversation = Conversation()

    system_message = Message(role="system", content="System prompt")
    conversation.add_message(system_message)

    user_message = Message(role="user", content="Hello")
    conversation.add_message(user_message)

    assistant_message = Message(role="assistant", content="Hi there")
    conversation.add_message(assistant_message)

    pretty_string = conversation.to_pretty_string()

    assert (
        pretty_string
        == """\033[1;36mSystem\033[0m
--------
System prompt

\033[1;32mUser\033[0m
------
Hello

\033[1;35mAssistant\033[0m
-----------
Hi there"""
    )


def test_conversation_to_markdown():
    conversation = Conversation()

    system_message = Message(role="system", content="System prompt")
    conversation.add_message(system_message)

    user_message = Message(role="user", content="Hello")
    conversation.add_message(user_message)

    assistant_message = Message(role="assistant", content="Hi there")
    conversation.add_message(assistant_message)

    markdown_string = conversation.to_markdown()

    assert (
        markdown_string
        == """## System 
System prompt

## User 
Hello

## Assistant 
Hi there"""
    )


def test_conversation_openai_messages():
    conversation = Conversation()

    system_message = Message(role="system", content="System prompt")
    conversation.add_message(system_message)

    user_message = Message(role="user", content="Hello")
    conversation.add_message(user_message)

    assistant_message = Message(role="assistant", content="Hi there")
    conversation.add_message(assistant_message)

    openai_messages = conversation.openai_messages

    assert len(openai_messages) == 3
    assert openai_messages[0] == {"role": "system", "content": "System prompt"}
    assert openai_messages[1] == {"role": "user", "content": "Hello"}
    assert openai_messages[2] == {"role": "assistant", "content": "Hi there"}


def test_conversation_from_prompts_user_prompt():
    user_prompt = "Hello"
    conversation = Conversation.from_prompts(user_prompt)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello"


def test_conversation_from_prompts_user_assistant_prompt():
    user_prompt = "Hello"
    assistant_prompt = "Hi there"
    conversation = Conversation.from_prompts(user_prompt, assistant_prompt)

    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "Hi there"


def test_conversation_from_prompts_user_assistant_system_prompt():
    user_prompt = "Hello"
    assistant_prompt = "Hi there"
    system_prompt = "System prompt"
    conversation = Conversation.from_prompts(
        user_prompt, assistant_prompt, system_prompt
    )

    assert len(conversation.messages) == 3
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == "System prompt"
    assert conversation.messages[1].role == "user"
    assert conversation.messages[1].content == "Hello"
    assert conversation.messages[2].role == "assistant"
    assert conversation.messages[2].content == "Hi there"


def create_text_completion():
    conversation = Conversation.from_prompts("Hello", "Hi there")

    return TextCompletion(
        conversation=conversation,
        model="gpt-4o",
        content="Test content",
        start_time=100.0,
        end_time=105.5,
        full_token_log_probs=[
            [
                TokenLogProb(token="Test", log_prob=-0.1),
                TokenLogProb(token="Example", log_prob=-0.2),
            ],
            [
                TokenLogProb(token="content", log_prob=-0.2),
                TokenLogProb(token="word", log_prob=-0.3),
            ],
        ],
    )


def create_structured_completion():
    conversation = Conversation.from_prompts("Hello", "Hi there")

    return StructuredCompletion(
        conversation=conversation,
        model="gpt-4o",
        content={"key": "value"},
        start_time=100.0,
        end_time=107.0,
        full_token_log_probs=[
            [
                TokenLogProb(token="{", log_prob=-0.1),
                TokenLogProb(token="[", log_prob=-0.3),
            ],
            [
                TokenLogProb(token="key", log_prob=-0.2),
                TokenLogProb(token="name", log_prob=-0.4),
            ],
        ],
    )


def test_text_completion_duration():
    text_completion = create_text_completion()

    assert text_completion.duration == 5.5


def test_text_completion_token_log_probs():
    text_completion = create_text_completion()

    assert len(text_completion.token_log_probs) == 2
    assert text_completion.token_log_probs[0].token == "Test"
    assert text_completion.token_log_probs[0].log_prob == -0.1
    assert text_completion.token_log_probs[1].token == "content"
    assert text_completion.token_log_probs[1].log_prob == -0.2


def test_text_completion_to_pretty_log_probs_string():
    text_completion = create_text_completion()

    pretty_log_probs_string = text_completion.to_pretty_log_probs_string()

    assert pretty_log_probs_string == "\x1b[1;92mTest\x1b[0m\x1b[1;92mcontent\x1b[0m"


def test_text_completion_to_pretty_log_probs_string_with_probabilities():
    text_completion = create_text_completion()

    pretty_log_probs_string = text_completion.to_pretty_log_probs_string(
        show_probabilities=True
    )

    assert (
        pretty_log_probs_string
        == "\x1b[1;92mTest[0.9]\x1b[0m\x1b[1;92mcontent[0.82]\x1b[0m"
    )


def test_structured_completion_duration():
    structured_completion = create_structured_completion()

    assert structured_completion.duration == 7.0


def test_structured_completion_token_log_probs():
    structured_completion = create_structured_completion()

    assert len(structured_completion.token_log_probs) == 2
    assert structured_completion.token_log_probs[0].token == "{"
    assert structured_completion.token_log_probs[0].log_prob == -0.1
    assert structured_completion.token_log_probs[1].token == "key"
    assert structured_completion.token_log_probs[1].log_prob == -0.2


def test_structured_completion_to_pretty_log_probs_string():
    structured_completion = create_structured_completion()

    pretty_log_probs_string = structured_completion.to_pretty_log_probs_string()

    assert pretty_log_probs_string == "\x1b[1;92m{\x1b[0m\x1b[1;92mkey\x1b[0m"


def test_structured_completion_to_pretty_log_probs_string_with_probabilities():
    structured_completion = create_structured_completion()

    pretty_log_probs_string = structured_completion.to_pretty_log_probs_string(
        show_probabilities=True
    )

    assert (
        pretty_log_probs_string == "\x1b[1;92m{[0.9]\x1b[0m\x1b[1;92mkey[0.82]\x1b[0m"
    )
