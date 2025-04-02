from limin import Conversation, Message


example_conversation_wrong = Conversation(messages=[
    Message(role="user", content="Hello, how are you?"),
    Message(role="assistant", content="I'm good, tank you!"),
])

example_conversation_correct = Conversation(messages=[
    Message(role="user", content="Hello, how are you?"),
    Message(role="assistant", content="I'm good, thank you!"),
])