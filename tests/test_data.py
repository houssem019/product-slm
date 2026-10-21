import json
from pathlib import Path

import pytest

from pslm.data.labels import SPLITS, check_split, load_titles, parse_tsv_line, write_jsonl
from pslm.data.review import apply_review, review
from pslm.schema import ProductAttributes


def test_schema_normalizes_blanks_and_material_case():
    a = ProductAttributes(category="apparel", brand="  ", size="null", material="Stainless  Steel")
    assert (a.brand, a.size, a.material) == (None, None, "stainless steel")


def test_schema_rejects_unknown_category_and_extra_fields():
    with pytest.raises(ValueError):
        ProductAttributes(category="clothing", brand=None, size=None, material=None)
    with pytest.raises(ValueError):
        ProductAttributes(category="apparel", brand=None, size=None, material=None, color="red")


def test_parse_tsv_line_dash_is_null():
    idx, a = parse_tsv_line("7\telectronics\tAGVEE\t1.5ft\t-")
    assert idx == 7
    assert a.model_dump() == {"category": "electronics", "brand": "AGVEE", "size": "1.5ft", "material": None}


def _write_split(tmp: Path, title: str, label: dict) -> tuple[Path, Path]:
    titles, labels = tmp / "titles", tmp / "labels"
    write_jsonl(titles / "test.jsonl", [{"id": "A1", "title": title, "source_category": "x"}])
    write_jsonl(labels / "test.jsonl", [{"id": "A1", "title": title, "label": label}])
    return titles, labels


def test_check_flags_ungrounded_brand_and_size(tmp_path):
    label = {"category": "home_kitchen", "brand": "Thermos", "size": "20 oz", "material": None}
    titles, labels = _write_split(tmp_path, "Stainless Bottle 24oz", label)
    rep = check_split("test", titles, labels)
    assert len(rep.errors) == 2


def test_check_grounding_ignores_case_and_spaces(tmp_path):
    label = {"category": "home_kitchen", "brand": "thermos", "size": "24 OZ", "material": None}
    titles, labels = _write_split(tmp_path, "THERMOS Bottle 24oz", label)
    assert check_split("test", titles, labels).ok


def test_apply_review_keeps_teacher_label_only_when_changed():
    rec = {"id": "A1", "label": {"category": "apparel", "brand": None, "size": None, "material": None}}
    accepted = apply_review(rec, None, "me")
    assert accepted["reviewed"] and "teacher_label" not in accepted
    edited = apply_review(rec, {**rec["label"], "size": "XL"}, "me")
    assert edited["label"]["size"] == "XL" and edited["teacher_label"] == rec["label"]


def test_review_loop_saves_progress(tmp_path):
    label = {"category": "apparel", "brand": None, "size": None, "material": None}
    write_jsonl(tmp_path / "test.jsonl", [{"id": "A", "title": "t", "label": label}] * 2)
    # accept the first record, then edit the second: keep category, set brand, keep size, keep material
    answers = iter(["", "e", "", "Acme", "", ""])
    assert review("test", "me", ask=lambda _: next(answers), root=tmp_path) == 2
    rows = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert all(r["reviewed"] for r in rows)
    assert rows[1]["label"]["brand"] == "Acme" and rows[1]["teacher_label"]["brand"] is None


@pytest.mark.parametrize("split", SPLITS)
def test_committed_labels_are_valid_and_grounded(split):
    rep = check_split(split)
    assert rep.n_labels == rep.n_titles, f"{split}: not every title is labeled"
    assert rep.ok, rep.errors[:5]


def test_splits_are_disjoint():
    ids = {s: {r["id"] for r in load_titles(s)} for s in SPLITS}
    assert (
        not (ids["train"] & ids["test"])
        and not (ids["train"] & ids["val"])
        and not (ids["val"] & ids["test"])
    )
