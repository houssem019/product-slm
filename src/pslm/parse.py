"""Turn raw model text into ProductAttributes. Shared by the teacher labeler, the baselines and the SLM eval, so
"JSON validity" means the same thing for every model."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from pslm.schema import ProductAttributes


@dataclass
class Parsed:
    attrs: ProductAttributes | None
    error: str | None = None  # not_json | invalid_schema | refusal | truncated

    @property
    def ok(self) -> bool:
        return self.attrs is not None


def _json_object(text: str) -> str:
    """Strip anything around the outermost {...}: code fences, or a preamble from models without constrained
    decoding. A model that *only* needed this to parse still counts as valid, as it would in a lenient
    production parser. Strict validity is measured separately (`strict_ok`)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


def parse_output(text: str, stop_reason: str = "end_turn") -> Parsed:
    if stop_reason == "refusal":
        return Parsed(None, "refusal")
    try:
        data = json.loads(_json_object(text))
    except json.JSONDecodeError:
        return Parsed(None, "truncated" if stop_reason == "max_tokens" else "not_json")
    try:
        return Parsed(ProductAttributes.model_validate(data))
    except ValidationError:
        return Parsed(None, "invalid_schema")


def strict_ok(text: str) -> bool:
    """True only if the whole text is the JSON object, with nothing around it."""
    try:
        ProductAttributes.model_validate(json.loads(text))
        return True
    except (json.JSONDecodeError, ValidationError):
        return False
