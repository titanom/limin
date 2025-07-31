from limin import Tool, generate_tool_call_completion
from pydantic import BaseModel, Field


def get_weather(location: str) -> str:
    if location == "Tokyo":
        return "The weather in Tokyo is sunny"
    elif location == "New York":
        return "The weather in New York is cloudy"
    elif location == "Berlin":
        return "The weather in Berlin is shitty"
    else:
        return f"Sorry, I don't know the weather in {location}"


class GetWeatherParameters(BaseModel):
    location: str = Field(description="The location to get the weather for")


get_weather_tool = Tool(
    name="get_weather",
    description="Get the current temperature in the city",
    parameters=GetWeatherParameters,
)


class CalculatorParameters(BaseModel):
    operation: str = Field(description="The operation to perform (+, -, *, /)")
    numbers: list[int] = Field(description="The numbers to perform the operation on")


calculator_tool = Tool(
    name="calculator",
    description="Perform a calculation",
    parameters=CalculatorParameters,
)


async def main():
    completion = await generate_tool_call_completion(
        "What is the purpose of life? Do not call any tools.",
        [get_weather_tool, calculator_tool],
    )
    tool_call = completion.tool_calls[0]
    print(tool_call)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
