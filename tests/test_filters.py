"""Tests for cohort filtering."""

from moirai.schema import Run, Step, Result
from moirai.filters import filter_runs


def _make_run(run_id: str, model: str = "m1", harness: str = "h1",
              task_family: str = "tf1", tags: dict | None = None) -> Run:
    return Run(
        run_id=run_id,
        task_id="t1",
        model=model,
        harness=harness,
        task_family=task_family,
        tags=tags or {},
        steps=[Step(idx=0, type="llm", name="plan")],
        result=Result(success=True),
    )


def test_filter_by_model():
    runs = [_make_run("r1", model="a"), _make_run("r2", model="b")]
    assert len(filter_runs(runs, model="a")) == 1
    assert filter_runs(runs, model="a")[0].run_id == "r1"


def test_filter_by_harness():
    runs = [_make_run("r1", harness="x"), _make_run("r2", harness="y")]
    assert len(filter_runs(runs, harness="x")) == 1


def test_filter_by_task_family():
    runs = [_make_run("r1", task_family="bug"), _make_run("r2", task_family="feat")]
    assert len(filter_runs(runs, task_family="bug")) == 1


def test_filter_by_tag():
    runs = [
        _make_run("r1", tags={"cohort": "baseline"}),
        _make_run("r2", tags={"cohort": "experiment"}),
    ]
    assert len(filter_runs(runs, tags={"cohort": "baseline"})) == 1


def test_filter_and_logic():
    runs = [
        _make_run("r1", model="a", harness="x"),
        _make_run("r2", model="a", harness="y"),
        _make_run("r3", model="b", harness="x"),
    ]
    result = filter_runs(runs, model="a", harness="x")
    assert len(result) == 1
    assert result[0].run_id == "r1"


def test_filter_no_matches():
    runs = [_make_run("r1", model="a")]
    assert filter_runs(runs, model="nonexistent") == []


def test_filter_numeric_tag_coercion():
    runs = [_make_run("r1", tags={"retries": 3})]
    assert len(filter_runs(runs, tags={"retries": "3"})) == 1


def test_filter_no_filters_returns_all():
    runs = [_make_run("r1"), _make_run("r2")]
    assert len(filter_runs(runs)) == 2
