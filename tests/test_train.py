import json

import pytest

from pslm.data.labels import load_labels
from pslm.prompts import STUDENT_SYSTEM
from pslm.schema import FIELDS, ProductAttributes
from pslm.train.format import build, target_json, to_messages


def test_target_json_is_canonical_and_parses_back():
    label = {"material": None, "size": "XL", "brand": "Acme", "category": "apparel"}  # scrambled key order
    out = target_json(label)
    assert list(json.loads(out)) == list(FIELDS)
    assert out == '{"category": "apparel", "brand": "Acme", "size": "XL", "material": null}'
    ProductAttributes.model_validate_json(out)


def test_messages_use_the_student_prompt():
    msgs = to_messages("Acme tee", {"category": "apparel", "brand": None, "size": None, "material": None})
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == STUDENT_SYSTEM and msgs[1]["content"] == "Title: Acme tee"


def test_test_split_cannot_be_exported_for_training():
    with pytest.raises(ValueError):
        build("test")
    assert len(build("train")) == len(load_labels("train"))


def test_loss_mask_covers_only_the_answer():
    """With the real Qwen2.5 chat template: the unmasked tokens decode to exactly the JSON + end-of-turn."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from pslm.train.train import IGNORE, Collator, tokenize

    try:
        tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    except OSError:
        pytest.skip("tokenizer not cached and no network")
    label = {"category": "apparel", "brand": "Acme", "size": "XL", "material": "cotton"}
    ex = tokenize(tok, to_messages("Acme XL cotton tee", label), max_len=384)
    answer = [t for t, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != IGNORE]
    assert tok.decode(answer).strip() == target_json(label) + "<|im_end|>"
    assert ex["labels"][0] == IGNORE  # the system prompt is masked

    batch = Collator(tok.pad_token_id)([ex, tokenize(tok, to_messages("x", label), max_len=384)])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert (batch["labels"][1][batch["attention_mask"][1] == 0] == IGNORE).all()  # padding never trained on

    with pytest.raises(ValueError, match="max_len"):
        tokenize(tok, to_messages("x", label), max_len=10)
