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


class GetCalculatorParameters(BaseModel):
    operation: str = Field(
        description="Operation to perform e.g. add, subtract, multiply, divide"
    )
    numbers: list[int] = Field(
        description="List of numbers to perform the operation on"
    )


get_calculator_tool = Tool(
    name="get_calculator",
    description="Get the result of the operation on the list of numbers",
    parameters=GetCalculatorParameters,
)


async def main():
    completion_weather = await generate_tool_call_completion(
        "What's the weather like in Paris today?",
        [get_weather_tool, get_calculator_tool],
    )
    print(completion_weather)

    completion_calculator = await generate_tool_call_completion(
        "What's the result of 2+2?",
        get_calculator_tool,
    )
    print(completion_calculator)


if __name__ == "__main__":
    asyncio.run(main())
