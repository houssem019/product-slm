"""Runtime settings from the environment (with or without the PSLM_ prefix) or a .env file."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default):
    return Field(default, validation_alias=AliasChoices(f"PSLM_{name}", name))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = _env("LLM_PROVIDER", "fake")  # anthropic | ollama | fake
    # The teacher: labels are only as good as this model, and labeling is a one-off batch job.
    teacher_model: str = _env("TEACHER_MODEL", "claude-opus-5-5")
    llm_effort: str = _env("LLM_EFFORT", "low")  # extraction is easy; thinking tokens are the main cost
    ollama_model: str = _env("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    cassette_dir: Path = _env("CASSETTE_DIR", ROOT / "data" / "cassettes")


settings = Settings()
