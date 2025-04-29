import pytest
from typing import Literal, cast
from openai.types.chat import ChatCompletionMessageParam

from limin import Conversation, Message, TokenLogProb


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
