from __future__ import annotations

from typing import Any

from moirai.schema import (
    KNOWN_STEP_TYPES,
    STEP_TYPE_ALIASES,
    Result,
    Run,
    Step,
)


def _normalize_step_type(raw_type: str) -> str:
    t = raw_type.lower().strip()
    if t in STEP_TYPE_ALIASES:
        return STEP_TYPE_ALIASES[t]
    if t in KNOWN_STEP_TYPES:
        return t
    return "other"


def _coerce_metric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


RECOGNIZED_METRICS = {"tokens_in", "tokens_out", "latency_ms"}


def _normalize_metrics(raw: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in raw.items():
        coerced = _coerce_metric(value)
        if coerced is not None:
            result[key] = coerced
    return result


def _normalize_step(raw: dict[str, Any], idx_fallback: int, warnings: list[str]) -> Step:
    idx = raw.get("idx")
    if idx is None:
        idx = idx_fallback
        warnings.append(f"step {idx_fallback}: missing idx, assigned {idx_fallback}")
    elif not isinstance(idx, int):
        try:
            idx = int(idx)
        except (ValueError, TypeError):
            idx = idx_fallback
            warnings.append(f"step {idx_fallback}: non-integer idx, assigned {idx_fallback}")

    raw_type = raw.get("type", "other")
    step_type = _normalize_step_type(raw_type)
    if raw_type.lower().strip() not in KNOWN_STEP_TYPES and raw_type.lower().strip() not in STEP_TYPE_ALIASES:
        warnings.append(f"step {idx}: unknown type '{raw_type}', normalized to 'other'")

    name = raw.get("name", "unnamed")
    status = raw.get("status", "ok")
    input_data = raw.get("input", {})
    output_data = raw.get("output", {})
    metrics = _normalize_metrics(raw.get("metrics", {}))
    attrs = raw.get("attrs", {})

    return Step(
        idx=idx,
        type=step_type,
        name=name,
        status=status,
        input=input_data if isinstance(input_data, dict) else {},
        output=output_data if isinstance(output_data, dict) else {},
        metrics=metrics,
        attrs=attrs if isinstance(attrs, dict) else {},
    )


def _normalize_result(raw: dict[str, Any] | None, steps: list[Step], strict: bool) -> Result:
    if raw is None or not isinstance(raw, dict):
        if strict:
            raise ValueError("missing result object (strict mode)")
        has_error_step = any(s.type == "error" for s in steps)
        if has_error_step:
            return Result(success=False)
        return Result(success=None)

    success = raw.get("success")
    if success is None and strict:
        raise ValueError("result.success is null (strict mode)")

    if success is not None:
        success = bool(success)

    return Result(
        success=success,
        score=raw.get("score"),
        label=raw.get("label"),
        error_type=raw.get("error_type"),
        summary=raw.get("summary"),
    )


def normalize_run(raw: dict[str, Any], strict: bool = False) -> tuple[Run, list[str]]:
    """Normalize a raw dict into a Run. Returns (run, warnings)."""
    warnings: list[str] = []

    run_id = raw.get("run_id")
    task_id = raw.get("task_id")
    if not run_id:
        raise ValueError("missing required field: run_id")
    if not task_id:
        raise ValueError("missing required field: task_id")

    raw_steps = raw.get("steps")
    if raw_steps is None:
        raise ValueError("missing required field: steps")
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list")

    if len(raw_steps) == 0:
        warnings.append("steps list is empty")

    # Normalize steps
    steps = [_normalize_step(s, i, warnings) for i, s in enumerate(raw_steps)]

    # Check for duplicate idx values
    seen_idx: dict[int, int] = {}
    for i, step in enumerate(steps):
        if step.idx in seen_idx:
            warnings.append(f"duplicate idx {step.idx} at positions {seen_idx[step.idx]} and {i}")
        else:
            seen_idx[step.idx] = i

    # Stable sort by idx
    steps.sort(key=lambda s: s.idx)

    # Normalize result
    result = _normalize_result(raw.get("result"), steps, strict)

    return Run(
        run_id=str(run_id),
        task_id=str(task_id),
        task_family=raw.get("task_family"),
        agent=raw.get("agent"),
        model=raw.get("model"),
        harness=raw.get("harness"),
        timestamp=raw.get("timestamp"),
        tags=raw.get("tags", {}),
        steps=steps,
        result=result,
    ), warnings
