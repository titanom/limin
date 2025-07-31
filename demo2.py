"""
Minimal ‘calculator’ tool-calling loop that follows the current
OpenAI function-/tool-calling spec (openai-python ≥ 1.0).

  pip install --upgrade openai
  export OPENAI_API_KEY=sk-…
"""

from __future__ import annotations
import json
import os
from openai import OpenAI

# ── 0 · client ────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1 · describe the tool ─────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math to compute, e.g. '2*(3+4)'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


# ── 2 · local implementation of the tool ─────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))  # ⚠ never use eval in prod
    except Exception as exc:
        return f"error: {exc}"


# ── 3 · first request: let GPT decide if it wants the tool ────────────────────
messages = [{"role": "user", "content": "What is (18 + 24) / 7 ?"}]

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # “auto” per spec
)

assistant_msg = resp.choices[0].message
messages.append(assistant_msg)  # keep the tool-call message intact
print(type(assistant_msg.tool_calls[0]))
print(assistant_msg.tool_calls[0])

# ── 4 · run any tool calls & append *only* a role="tool" message --------------
if assistant_msg.tool_calls:
    for call in assistant_msg.tool_calls:
        if call.function.name == "calculator":
            args = json.loads(call.function.arguments)
            result = calculator(**args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,  # mandatory link back to the call
                    "content": result,
                    # "name": "calculator"     # optional but harmless
                }
            )

    # ── 5 · second request: GPT crafts the final answer ───────────────────────
    print(messages)
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    print(final.choices[0].message.content)
else:
    # GPT chose not to call a tool
    print(assistant_msg.content)
