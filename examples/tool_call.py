import asyncio

from pydantic import BaseModel, Field

from limin import Tool, generate_tool_call_completion


class GetWeatherParameters(BaseModel):
    location: str = Field(description="City and country e.g. Bogotá, Colombia")


get_weather_tool = Tool(
    name="get_weather",
    description="Get current temperature for provided location in celsius.",
    parameters=GetWeatherParameters,
)


async def main():
    completion = await generate_tool_call_completion(
        "What's the weather like in Paris today?",
        get_weather_tool,
    )
    print(completion)


if __name__ == "__main__":
    asyncio.run(main())
