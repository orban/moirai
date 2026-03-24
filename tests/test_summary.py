"""Tests for summary statistics."""

from moirai.schema import Step, Result, Run
from moirai.analyze.summary import summarize_runs


def _make_run(run_id: str, success: bool | None = True, steps: list[Step] | None = None,
              error_type: str | None = None) -> Run:
    if steps is None:
        steps = [
            Step(idx=0, type="llm", name="plan", metrics={"tokens_in": 100.0, "tokens_out": 50.0, "latency_ms": 1000.0}),
            Step(idx=1, type="tool", name="search", metrics={"tokens_in": 200.0, "tokens_out": 80.0, "latency_ms": 2000.0}),
        ]
    return Run(
        run_id=run_id,
        task_id="t1",
        steps=steps,
        result=Result(success=success, error_type=error_type),
    )


def test_basic_summary():
    runs = [_make_run("r1", True), _make_run("r2", True), _make_run("r3", False)]
    s = summarize_runs(runs)
    assert s.run_count == 3
    assert s.success_rate is not None
    assert abs(s.success_rate - 2 / 3) < 0.01
    assert s.avg_steps == 2.0
    assert s.median_steps == 2.0


def test_all_null_success():
    runs = [_make_run("r1", success=None), _make_run("r2", success=None)]
    s = summarize_runs(runs)
    assert s.success_rate is None


def test_mixed_null_success():
    runs = [_make_run("r1", True), _make_run("r2", None), _make_run("r3", False)]
    s = summarize_runs(runs)
    # Only r1 and r3 count: 1/2 = 0.5
    assert s.success_rate is not None
    assert abs(s.success_rate - 0.5) < 0.01


def test_no_metrics():
    steps = [Step(idx=0, type="llm", name="plan")]
    runs = [_make_run("r1", steps=steps)]
    s = summarize_runs(runs)
    assert s.avg_tokens_in is None
    assert s.avg_tokens_out is None
    assert s.avg_latency_ms is None


def test_empty_runs():
    s = summarize_runs([])
    assert s.run_count == 0
    assert s.success_rate is None
    assert s.avg_steps == 0.0


def test_error_counts():
    runs = [
        _make_run("r1", False, error_type="tool_timeout"),
        _make_run("r2", False, error_type="tool_timeout"),
        _make_run("r3", False, error_type="loop_detected"),
    ]
    s = summarize_runs(runs)
    assert s.error_counts == {"tool_timeout": 2, "loop_detected": 1}


def test_top_signatures():
    runs = [_make_run("r1"), _make_run("r2"), _make_run("r3")]
    s = summarize_runs(runs)
    assert len(s.top_signatures) == 1  # all same signature
    assert s.top_signatures[0][1] == 3  # count = 3
