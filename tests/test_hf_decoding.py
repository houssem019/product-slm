"""The HF backend must decode greedily with no repetition penalty, whatever the model's shipped defaults say."""

from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")

from pslm.llm.hf_client import greedy_config  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def test_greedy_config_drops_shipped_repetition_penalty_and_sampling():
    shipped = transformers.GenerationConfig(
        do_sample=True,
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        eos_token_id=[1, 2],
        bos_token_id=0,
    )
    cfg = greedy_config(shipped, SimpleNamespace(pad_token_id=5))
    assert cfg.repetition_penalty == 1.0 and cfg.do_sample is False
    assert cfg.eos_token_id == [1, 2] and cfg.pad_token_id == 5


def test_generate_matches_a_manual_argmax_loop_even_if_the_model_ships_a_penalty():
    """Behavioral check: with Qwen's real penalty planted in the config, generate() must still equal plain argmax."""
    import torch

    try:
        tok = transformers.AutoTokenizer.from_pretrained(TINY)
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY).eval()
    except OSError:
        pytest.skip("tiny model not cached and no network")
    model.generation_config.repetition_penalty = 1.1  # what Qwen2.5-Instruct ships
    model.generation_config.do_sample = True
    model.generation_config = greedy_config(model.generation_config, tok)

    ids = tok("Title: Acme Acme Acme XL XL cotton tee", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, min_new_tokens=8)
        manual = ids
        for _ in range(8):
            nxt = model(manual).logits[:, -1].argmax(-1, keepdim=True)
            manual = torch.cat([manual, nxt], dim=1)
    assert out[0].tolist() == manual[0].tolist()
