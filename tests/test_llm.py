import json
from types import SimpleNamespace

import pytest

from pslm.data.labels import load_titles
from pslm.distill.batch import build_requests, estimate, rows_from_results
from pslm.distill.label import evaluate, predict, run_summary
from pslm.eval.metrics import score
from pslm.llm import get_llm, model_slug
from pslm.llm.anthropic_client import request_params
from pslm.llm.replay import CassetteMiss, ReplayLLM, ScriptedLLM
from pslm.parse import parse_output, strict_ok
from pslm.prompts import EXAMPLES, STUDENT_SYSTEM, teacher_system
from pslm.schema import Category, ProductAttributes, output_schema

GOOD = '{"category": "apparel", "brand": "Acme", "size": "XL", "material": "cotton"}'


# --- schema / parsing ---------------------------------------------------------------------------------------


def test_output_schema_is_flat_and_complete():
    s = output_schema()
    assert "$defs" not in json.dumps(s) and "$ref" not in json.dumps(s)
    assert s["required"] == ["category", "brand", "size", "material"] and s["additionalProperties"] is False
    assert s["properties"]["category"]["enum"] == [c.value for c in Category]


@pytest.mark.parametrize(
    ("text", "stop", "error"),
    [
        (GOOD, "end_turn", None),
        ("```json\n" + GOOD + "\n```", "end_turn", None),  # fenced: lenient parse succeeds
        ('{"category": "apparel"', "max_tokens", "truncated"),
        ("I can't tell.", "end_turn", "not_json"),
        (
            '{"category": "shoes", "brand": null, "size": null, "material": null}',
            "end_turn",
            "invalid_schema",
        ),
        ("", "refusal", "refusal"),
    ],
)
def test_parse_output(text, stop, error):
    assert parse_output(text, stop).error == error


def test_strict_json_rejects_fences():
    assert strict_ok(GOOD) and not strict_ok("```json\n" + GOOD + "\n```")


# --- prompts ------------------------------------------------------------------------------------------------


def test_teacher_prompt_embeds_guidelines_and_categories():
    p = teacher_system()
    assert "<guidelines>" in p and "Plating is a finish" in p
    assert all(c.value in p for c in Category) and all(c.value in STUDENT_SYSTEM for c in Category)


def test_few_shot_examples_come_from_train_only():
    held_out = {t["title"] for s in ("val", "test") for t in load_titles(s)}
    train = {t["title"] for t in load_titles("train")}
    for title, label in EXAMPLES:
        assert title in train and title not in held_out
        ProductAttributes.model_validate(label)


# --- metrics ------------------------------------------------------------------------------------------------


def _lab(category="apparel", brand=None, size=None, material=None):
    return {"category": category, "brand": brand, "size": size, "material": material}


def test_score_normalizes_and_counts_prf():
    gold = {"a": _lab(brand="Acme", size="20 oz"), "b": _lab(material="cotton"), "c": _lab()}
    preds = {
        "a": _lab(brand="ACME", size="20oz"),  # case + spacing differences still match
        "b": _lab(material="wool"),  # wrong value: FP and FN
        "c": _lab(size="XL"),  # hallucinated value: FP
    }
    rep = score(preds, gold)
    assert rep.correct == {"category": 3, "brand": 3, "size": 2, "material": 2}
    assert rep.exact == 1
    assert rep.prf["size"] == {"precision": 0.5, "recall": 1.0, "f1": 0.667}
    assert rep.prf["material"]["f1"] == 0.0


def test_score_invalid_output_counts_as_wrong_and_f1_undefined_without_positives():
    rep = score({"a": None}, {"a": _lab()}, {"a": "not_json"})
    assert rep.valid == 0 and rep.correct["category"] == 0 and rep.errors == {"not_json": 1}
    assert rep.prf["brand"]["f1"] is None


# --- providers ----------------------------------------------------------------------------------------------


def test_replay_records_then_replays(tmp_path):
    cassette = tmp_path / "m.json"
    rec = ReplayLLM(cassette, inner=ScriptedLLM([GOOD], model="m"))
    assert rec.complete("sys", "Title: x", output_schema()).text == GOOD
    replay = ReplayLLM(cassette)
    assert replay.complete("sys", "Title: x", output_schema()).text == GOOD
    assert replay.model == "m"
    with pytest.raises(CassetteMiss):
        replay.complete("changed system prompt", "Title: x", output_schema())


def test_get_llm_validates_provider(tmp_path):
    with pytest.raises(ValueError):
        get_llm("fake")
    with pytest.raises(ValueError):
        get_llm("openai")
    assert get_llm("fake", cassette=tmp_path / "x.json").model == "replay"


def test_anthropic_request_params_structured_output_and_cached_system():
    p = request_params("claude-opus-5-5", "SYS", "Title: x", output_schema(), "low")
    assert p["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": output_schema()},
    }
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "fallbacks" not in p and "thinking" not in p  # batch-safe; Opus 5.5 rejects disabling thinking


def test_model_slug():
    assert model_slug("qwen2.5:7b") == "qwen2.5-7b"


# --- labeler ------------------------------------------------------------------------------------------------


def test_predict_and_evaluate_with_scripted_llm():
    titles = [{"id": "a", "title": "Acme XL cotton tee"}, {"id": "b", "title": "something"}]
    llm = ScriptedLLM([GOOD, "not json"])
    rows = predict(llm, titles, "SYS")
    assert llm.calls[0]["user"] == "Title: Acme XL cotton tee"
    assert rows[0]["pred"]["brand"] == "Acme" and rows[1]["error"] == "not_json"
    gold = [
        {"id": "a", "label": json.loads(GOOD)},
        {"id": "b", "label": _lab()},
        {"id": "z", "label": _lab()},
    ]
    rep = evaluate(rows, gold)  # "z" wasn't predicted, so it isn't scored
    assert rep.n == 2 and rep.exact == 1 and rep.valid == 1
    s = run_summary(rows, "claude-haiku-5-5")
    assert s["n"] == 2 and s["cost_usd"] > 0


# --- batch --------------------------------------------------------------------------------------------------


def test_build_requests_uses_ids_as_custom_ids():
    titles = [{"id": "B0ABC", "title": "t1"}, {"id": "B0DEF", "title": "t2"}]
    reqs = build_requests(titles, "claude-haiku-5-5", "low")
    assert [r["custom_id"] for r in reqs] == ["B0ABC", "B0DEF"]
    assert reqs[0]["params"]["system"][0]["text"] == teacher_system()


def test_rows_from_results_reorders_and_handles_failures():
    titles = {"a": {"id": "a", "title": "t"}, "b": {"id": "b", "title": "u"}}
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    msg = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=GOOD)],
        stop_reason="end_turn",
        model="claude-haiku-5-5",
        usage=usage,
    )
    results = [  # results arrive in any order
        SimpleNamespace(custom_id="b", result=SimpleNamespace(type="expired")),
        SimpleNamespace(custom_id="a", result=SimpleNamespace(type="succeeded", message=msg)),
    ]
    rows = rows_from_results(results, titles)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["pred"]["brand"] == "Acme" and rows[1]["error"] == "batch_expired"


def test_estimate_is_positive_and_cache_helps():
    est = estimate([{"id": str(i), "title": "Acme cotton tee"} for i in range(100)], "claude-opus-5-5")
    assert 0 < est["usd_batch_cached"] < est["usd_batch_uncached"]
