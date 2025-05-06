import asyncio
from openai import OpenAI, pydantic_function_tool
import json

from pydantic import BaseModel, Field

from limin.tool_call import Tool, generate_tool_call_completion
from limin import Conversation, Message


class GetWeatherParameters(BaseModel):
    location: str = Field(description="City and country e.g. Bogotá, Colombia")


messages = Conversation(
    messages=[Message(role="user", content="What's the weather like in Paris today?")]
)

print(GetWeatherParameters.model_json_schema())


async def main():
    completion = await generate_tool_call_completion(
        messages,
        [
            Tool(
                name="get_weather",
                description="Get current temperature for provided coordinates in celsius.",
                parameters=GetWeatherParameters,
            )
        ],
    )
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
