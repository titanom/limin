from .base import Tool, ToolMessage, UserMessage, AssistantMessage
from .base import Conversation, ModelConfiguration
from .text_completion import generate_text_completion_for_conversation
from .tool_call import generate_tool_call_completion_for_conversation


class Agent:
    def __init__(self, tools: list[Tool], model_configuration: ModelConfiguration):
        self.conversation = Conversation()
        self.tools = tools
        self.model_configuration = model_configuration

    async def respond(self, user_message: str) -> None:
        self.conversation.add_message(UserMessage(content=user_message))

        if len(self.tools) > 0:
            tool_call_completion = await generate_tool_call_completion_for_conversation(
                self.conversation,
                tools=self.tools,
                model_configuration=self.model_configuration,
            )

            assistant_message = AssistantMessage(
                content=None,
                tool_calls=tool_call_completion.tool_calls,
            )
            self.conversation.add_message(assistant_message)

            for tool_call in tool_call_completion.tool_calls:
                # Get the tool with the correct ID
                tool = next(tool for tool in self.tools if tool.name == tool_call.name)

                # Execute the tool
                content = tool.execute(**tool_call.arguments)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call.id,
                )
                self.conversation.add_message(tool_message)

            # Generate a response from the model
            text_completion = await generate_text_completion_for_conversation(
                self.conversation,
                model_configuration=self.model_configuration,
            )

            self.conversation.add_message(text_completion.to_assistant_message())
        else:
            text_completion = await generate_text_completion_for_conversation(
                self.conversation,
                model_configuration=self.model_configuration,
            )

            self.conversation.add_message(text_completion.to_assistant_message())
