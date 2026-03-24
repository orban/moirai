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


NAMED_FIELDS = {"model", "harness", "task_family"}


def parse_kv_filter(kv: str) -> tuple[str, str]:
    """Parse a 'key=value' string into (key, value).

    Raises ValueError if the string doesn't contain '='.
    """
    if "=" not in kv:
        raise ValueError(f"invalid filter format '{kv}', expected key=value")
    key, value = kv.split("=", 1)
    if not key:
        raise ValueError(f"invalid filter format '{kv}', empty key")
    return key, value


def apply_kv_filters(runs: list[Run], kv_pairs: list[str]) -> list[Run]:
    """Apply a list of K=V filter strings to runs.

    Named fields (model, harness, task_family) are matched directly.
    Unknown keys are treated as tag filters.
    All filters are AND'd.
    """
    model = None
    harness = None
    task_family = None
    tags: dict[str, str] = {}

    for kv in kv_pairs:
        key, value = parse_kv_filter(kv)
        if key == "model":
            model = value
        elif key == "harness":
            harness = value
        elif key == "task_family":
            task_family = value
        else:
            tags[key] = value

    return filter_runs(runs, model=model, harness=harness, task_family=task_family,
                       tags=tags if tags else None)
