import math
from typing import Generic, Literal, TypeVar, cast
from openai.types.chat import ChatCompletionMessageParam

from pydantic import BaseModel, Field

from .base import AssistantMessage
