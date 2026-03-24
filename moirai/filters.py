from __future__ import annotations

from moirai.schema import Run


def filter_runs(
    runs: list[Run],
    model: str | None = None,
    harness: str | None = None,
    task_family: str | None = None,
    tags: dict[str, str] | None = None,
) -> list[Run]:
    """Filter runs by field values and/or tag key-value pairs. All filters are AND'd."""
    result: list[Run] = []
    for run in runs:
        if model and run.model != model:
            continue
        if harness and run.harness != harness:
            continue
        if task_family and run.task_family != task_family:
            continue
        if tags:
            match = True
            for key, value in tags.items():
                tag_val = run.tags.get(key)
                if str(tag_val) != str(value):
                    match = False
                    break
            if not match:
                continue
        result.append(run)
    return result
