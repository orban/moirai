from __future__ import annotations

import json
from pathlib import Path

from moirai.normalize import normalize_run
from moirai.schema import Run, ValidationResult


def validate_file(file_path: Path, strict: bool = False) -> ValidationResult:
    """Validate a single JSON run file. Returns a ValidationResult."""
    warnings: list[str] = []
    errors: list[str] = []

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as e:
        return ValidationResult(file_path=str(file_path), passed=False, errors=[f"cannot read file: {e}"])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return ValidationResult(file_path=str(file_path), passed=False, errors=[f"invalid JSON: {e}"])

    if isinstance(data, list):
        return ValidationResult(
            file_path=str(file_path),
            passed=False,
            errors=["expected a single Run object, got a JSON array"],
        )

    if not isinstance(data, dict):
        return ValidationResult(
            file_path=str(file_path),
            passed=False,
            errors=[f"expected a JSON object, got {type(data).__name__}"],
        )

    # Check required fields
    for field in ("run_id", "task_id", "steps", "result"):
        if field not in data:
            errors.append(f"missing required field: {field}")

    if errors:
        return ValidationResult(file_path=str(file_path), passed=False, warnings=warnings, errors=errors)

    # Try normalization to catch deeper issues
    try:
        _, norm_warnings = normalize_run(data, strict=strict)
        warnings.extend(norm_warnings)
    except ValueError as e:
        if strict:
            return ValidationResult(file_path=str(file_path), passed=False, warnings=warnings, errors=[str(e)])
        warnings.append(str(e))

    passed = len(errors) == 0
    if strict and warnings:
        passed = False
        errors.extend(warnings)
        warnings = []

    return ValidationResult(file_path=str(file_path), passed=passed, warnings=warnings, errors=errors)


def _find_json_files(path: Path) -> list[Path]:
    """Recursively find .json files, skipping symlinks."""
    if path.is_file():
        if path.suffix == ".json":
            return [path]
        return []

    results: list[Path] = []
    for p in sorted(path.rglob("*.json")):
        if p.is_symlink():
            continue
        if p.is_file():
            results.append(p)
    return results


def load_runs(path: str | Path, strict: bool = False) -> tuple[list[Run], list[str]]:
    """Load and normalize runs from a file or directory.

    Returns (runs, warnings). Raises FileNotFoundError if path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")

    json_files = _find_json_files(path)
    if not json_files:
        return [], [f"no JSON files found in {path}"]

    runs: list[Run] = []
    warnings: list[str] = []
    seen_ids: dict[str, str] = {}  # run_id -> first file path

    for file_path in json_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            msg = f"{file_path}: skipped ({e})"
            if strict:
                raise ValueError(msg) from e
            warnings.append(msg)
            continue

        if isinstance(data, list):
            msg = f"{file_path}: expected a single Run object, got a JSON array"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue

        if not isinstance(data, dict):
            msg = f"{file_path}: expected a JSON object, got {type(data).__name__}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue

        try:
            run, norm_warnings = normalize_run(data, strict=strict)
        except ValueError as e:
            msg = f"{file_path}: {e}"
            if strict:
                raise ValueError(msg) from e
            warnings.append(msg)
            continue

        for w in norm_warnings:
            warnings.append(f"{file_path}: {w}")

        # Duplicate run_id check
        if run.run_id in seen_ids:
            warnings.append(
                f"{file_path}: duplicate run_id '{run.run_id}' "
                f"(first seen in {seen_ids[run.run_id]}), skipping"
            )
            continue

        seen_ids[run.run_id] = str(file_path)
        runs.append(run)

    return runs, warnings
