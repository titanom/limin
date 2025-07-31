import asyncio

from pydantic import BaseModel, Field

from limin import Tool, Agent, ModelConfiguration


class GetWeatherParameters(BaseModel):
    location: str = Field(description="City and country e.g. Bogotá, Colombia")


def get_weather_exec(location: str) -> str:
    # Mock weather function - in reality this would call a weather API
    return f"The current temperature in {location} is 22°C with partly cloudy skies."


get_weather_tool = Tool(
    name="get_weather",
    description="Get current temperature for provided location in celsius.",
    parameters=GetWeatherParameters,
    exec_fn=get_weather_exec,
)


class GetCalculatorParameters(BaseModel):
    operation: str = Field(
        description="Operation to perform e.g. add, subtract, multiply, divide"
    )
    numbers: list[int] = Field(
        description="List of numbers to perform the operation on"
    )


def get_calculator_exec(operation: str, numbers: list[int]) -> str:
    if operation == "add":
        result = sum(numbers)
    elif operation == "subtract":
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
    elif operation == "multiply":
        result = 1
        for num in numbers:
            result *= num
    elif operation == "divide":
        result = numbers[0]
        for num in numbers[1:]:
            if num == 0:
                return "Error: Division by zero"
            result /= num
    else:
        return f"Error: Unknown operation '{operation}'"

    return str(result)


get_calculator_tool = Tool(
    name="get_calculator",
    description="Get the result of the operation on the list of numbers",
    parameters=GetCalculatorParameters,
    exec_fn=get_calculator_exec,
)


async def main():
    model_config = ModelConfiguration()
    agent = Agent(
        tools=[get_weather_tool, get_calculator_tool],
        model_configuration=model_config,
    )

    await agent.respond("What's the weather like in Paris today?")
    print("Agent conversation after weather query:")
    for message in agent.conversation.messages:
        print(f"{message.role}: {message.content}")

    print("\n" + "=" * 50 + "\n")

    await agent.respond("What's the result of 2+2?")
    print("Agent conversation after calculator query:")
    for message in agent.conversation.messages:
        print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
