from pydantic import BaseModel


class ModelConfiguration(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 1.0
    log_probs: bool = False
    top_log_probs: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_p: float | None = None
    seed: int | None = None
    api_key: str | None = None
    base_url: str | None = None


DEFAULT_MODEL_CONFIGURATION = ModelConfiguration()
