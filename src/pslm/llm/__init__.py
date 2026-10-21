from pathlib import Path

from pslm.config import Settings, settings
from pslm.llm.base import LLMClient, LLMResponse, Usage

__all__ = ["LLMClient", "LLMResponse", "Usage", "get_llm", "model_slug"]


def model_slug(model: str) -> str:
    """File-name-safe model id, e.g. qwen2.5:7b -> qwen2.5-7b."""
    return model.replace(":", "-").replace("/", "-")


def get_llm(
    provider: str | None = None,
    cfg: Settings = settings,
    cassette: Path | None = None,
    record: bool = False,
    model: str | None = None,
) -> LLMClient:
    """Build an LLM. `fake` replays `cassette`; record=True wraps the real provider and saves to `cassette`."""
    from pslm.llm.replay import ReplayLLM

    provider = provider or cfg.llm_provider
    if provider == "fake":
        if cassette is None:
            raise ValueError("the fake provider needs a cassette to replay")
        return ReplayLLM(cassette)
    if provider == "anthropic":
        from pslm.llm.anthropic_client import AnthropicLLM

        llm: LLMClient = AnthropicLLM(model or cfg.teacher_model, effort=cfg.llm_effort)
    elif provider == "ollama":
        from pslm.llm.ollama_client import OllamaLLM

        llm = OllamaLLM(model or cfg.ollama_model, cfg.ollama_host)
    else:
        raise ValueError(f"unknown LLM provider {provider!r} (anthropic | ollama | fake)")
    return ReplayLLM(cassette, inner=llm) if record and cassette else llm
