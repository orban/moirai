"""Tests for schema types, sequence extraction, and normalization."""

from moirai.schema import Step, Result, Run, step_type_sequence, step_name_sequence, signature


# --- Schema basics ---

def test_step_defaults():
    s = Step(idx=0, type="llm", name="plan")
    assert s.status == "ok"
    assert s.input == {}
    assert s.output == {}
    assert s.metrics == {}
    assert s.attrs == {}


def test_result_defaults():
    r = Result(success=True)
    assert r.score is None
    assert r.label is None
    assert r.error_type is None
    assert r.summary is None


def test_run_defaults():
    r = Run(run_id="r1", task_id="t1")
    assert r.task_family is None
    assert r.steps == []
    assert r.result.success is None
    assert r.tags == {}


# --- Sequence extraction ---

def _make_run(step_specs: list[tuple[str, str]], success: bool = True) -> Run:
    steps = [Step(idx=i, type=t, name=n) for i, (t, n) in enumerate(step_specs)]
    return Run(run_id="r1", task_id="t1", steps=steps, result=Result(success=success))


def test_step_type_sequence():
    run = _make_run([("llm", "plan"), ("tool", "search"), ("llm", "reason")])
    assert step_type_sequence(run) == ["llm", "tool", "llm"]


def test_step_name_sequence():
    run = _make_run([("llm", "plan"), ("tool", "search"), ("llm", "reason")])
    assert step_name_sequence(run) == ["plan", "search", "reason"]


def test_signature():
    run = _make_run([("llm", "plan"), ("tool", "search"), ("judge", "verify")])
    assert signature(run) == "llm:plan > tool:search > judge:verify"


def test_signature_empty_steps():
    run = _make_run([])
    assert signature(run) == ""


# --- Normalization ---

from moirai.normalize import normalize_run


def test_normalize_basic():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"idx": 0, "type": "llm", "name": "plan"},
            {"idx": 1, "type": "tool", "name": "search"},
        ],
        "result": {"success": True},
    }
    run, warnings = normalize_run(raw)
    assert run.run_id == "r1"
    assert len(run.steps) == 2
    assert run.result.success is True
    assert warnings == []


def test_normalize_type_aliases():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"idx": 0, "type": "assistant", "name": "plan"},
            {"idx": 1, "type": "function_call", "name": "search"},
            {"idx": 2, "type": "critic", "name": "eval"},
            {"idx": 3, "type": "compress", "name": "compact"},
        ],
        "result": {"success": True},
    }
    run, _ = normalize_run(raw)
    assert [s.type for s in run.steps] == ["llm", "tool", "judge", "compaction"]


def test_normalize_unknown_type():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "banana", "name": "x"}],
        "result": {"success": True},
    }
    run, warnings = normalize_run(raw)
    assert run.steps[0].type == "other"
    assert any("unknown type" in w for w in warnings)


def test_normalize_missing_idx():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"type": "llm", "name": "a"},
            {"type": "tool", "name": "b"},
        ],
        "result": {"success": True},
    }
    run, warnings = normalize_run(raw)
    assert run.steps[0].idx == 0
    assert run.steps[1].idx == 1
    assert any("missing idx" in w for w in warnings)


def test_normalize_duplicate_idx():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"idx": 0, "type": "llm", "name": "a"},
            {"idx": 0, "type": "tool", "name": "b"},
        ],
        "result": {"success": True},
    }
    run, warnings = normalize_run(raw)
    assert len(run.steps) == 2
    assert any("duplicate idx" in w for w in warnings)


def test_normalize_sorts_by_idx():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"idx": 2, "type": "judge", "name": "c"},
            {"idx": 0, "type": "llm", "name": "a"},
            {"idx": 1, "type": "tool", "name": "b"},
        ],
        "result": {"success": True},
    }
    run, _ = normalize_run(raw)
    assert [s.name for s in run.steps] == ["a", "b", "c"]


def test_normalize_empty_steps():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [],
        "result": {"success": True},
    }
    run, warnings = normalize_run(raw)
    assert run.steps == []
    assert any("empty" in w for w in warnings)


def test_normalize_default_status():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "a"}],
        "result": {"success": True},
    }
    run, _ = normalize_run(raw)
    assert run.steps[0].status == "ok"


def test_normalize_result_inference_error_step():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [
            {"idx": 0, "type": "llm", "name": "a"},
            {"idx": 1, "type": "error", "name": "crash"},
        ],
    }
    run, _ = normalize_run(raw, strict=False)
    assert run.result.success is False


def test_normalize_result_inference_no_error():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "a"}],
    }
    run, _ = normalize_run(raw, strict=False)
    assert run.result.success is None


def test_normalize_strict_rejects_null_success():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "a"}],
        "result": {"success": None},
    }
    import pytest
    with pytest.raises(ValueError, match="strict mode"):
        normalize_run(raw, strict=True)


def test_normalize_strict_rejects_missing_result():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "a"}],
    }
    import pytest
    with pytest.raises(ValueError, match="strict mode"):
        normalize_run(raw, strict=True)


def test_normalize_missing_run_id():
    import pytest
    raw = {"task_id": "t1", "steps": [], "result": {"success": True}}
    with pytest.raises(ValueError, match="run_id"):
        normalize_run(raw)


def test_normalize_metric_coercion():
    raw = {
        "run_id": "r1",
        "task_id": "t1",
        "steps": [{"idx": 0, "type": "llm", "name": "a", "metrics": {"tokens_in": 100, "tokens_out": 50.5}}],
        "result": {"success": True},
    }
    run, _ = normalize_run(raw)
    assert run.steps[0].metrics["tokens_in"] == 100.0
    assert isinstance(run.steps[0].metrics["tokens_in"], float)
    assert run.steps[0].metrics["tokens_out"] == 50.5
