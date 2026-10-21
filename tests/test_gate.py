import json

from pslm.eval.gate import BASELINE, check, gate


def test_check_flags_drops_including_nested_and_missing():
    metrics = {"exact_match": 0.6, "accuracy": {"category": 0.8}, "f1": {"size": None}}
    thresholds = {"exact_match": 0.61, "accuracy": {"category": 0.79}, "f1": {"size": 0.5}, "json_valid": 0.9}
    failures = check(metrics, thresholds)
    assert any(f.startswith("exact_match") for f in failures)
    assert any(f.startswith("f1.size") for f in failures)  # an undefined F1 never passes a threshold
    assert any(f.startswith("json_valid") for f in failures)  # a missing metric fails
    assert not any(f.startswith("accuracy.category") for f in failures)


def test_gate_requires_thresholds_for_the_run(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"m:val": {"exact_match": 0.5}}))
    assert gate("m", "val", {"exact_match": 0.6}, path) == []
    assert gate("m", "test", {"exact_match": 0.6}, path) == ["no thresholds for m:test in b.json"]


def test_committed_baseline_matches_committed_reports():
    """The thresholds must pass against the committed prediction reports (they were derived from them)."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key, thresholds in baseline.items():
        run, split = key.split(":")
        report = json.loads(
            (BASELINE.parents[1] / "preds" / split / f"{run}.report.json").read_text(encoding="utf-8")
        )
        assert check(report["metrics"], thresholds) == [], key


def test_paired_bootstrap_detects_only_clear_differences():
    from pslm.eval.compare import paired_bootstrap

    same = {str(i): {"exact": i % 2 == 0} for i in range(100)}
    better = {str(i): {"exact": True} for i in range(100)}
    r = paired_bootstrap(better, same, "exact", n_boot=2000)
    assert r["diff"] == 0.5 and r["ci95"][0] > 0 and r["only_a"] == 50 and r["only_b"] == 0
    tie = paired_bootstrap(same, same, "exact", n_boot=2000)
    assert tie["diff"] == 0 and tie["ci95"] == (0.0, 0.0)
