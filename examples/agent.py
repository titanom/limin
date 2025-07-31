import asyncio

from pydantic import BaseModel, Field

from limin import Tool, Agent, ModelConfiguration


class GetWeatherParameters(BaseModel):
    location: str = Field(description="City and country e.g. Munich, Germany")


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
        description="Operation to perform - with +, -, *, / (e.g. 2*2)"
    )
    operator1: int = Field(description="First number to perform the operation on")
    operator2: int = Field(description="Second number to perform the operation on")


def get_calculator_exec(operation: str, operator1: int, operator2: int) -> str:
    if operation == "+":
        return str(operator1 + operator2)
    elif operation == "-":
        return str(operator1 - operator2)
    elif operation == "*":
        return str(operator1 * operator2)
    elif operation == "/":
        if operator2 == 0:
            return "Error: Division by zero"
        return str(operator1 / operator2)
    else:
        return f"Error: Unknown operation '{operation}'"


get_calculator_tool = Tool(
    name="get_calculator",
    description="Get the result of the operation on the list of numbers",
    parameters=GetCalculatorParameters,
    exec_fn=get_calculator_exec,
)


async def main():
    model_configuration = ModelConfiguration(model="gpt-4o", temperature=1.0)
    agent = Agent(
        system_prompt="You are a helpful assistant.",
        tools=[get_weather_tool, get_calculator_tool],
        model_configuration=model_configuration,
    )

    messages = await agent.process("How are you?")
    print(f"{messages[-1].role}: {messages[-1].content}")

    messages = await agent.process("What's the weather like in Paris today?")
    print("Agent conversation after weather query:")
    for message in messages:
        print(f"{message.role}: {message.content}")

    print("\n" + "=" * 50 + "\n")

    messages = await agent.process("What's the result of 2+2?")
    print("Agent conversation after calculator query:")
    for message in messages:
        print(f"{message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
