#!/usr/bin/env python3
"""Generate synthetic moirai runs for regression diagnosis demos.

Two scenarios:
  prompt_regression — testing frequency drops but pass rate stays the same
  timeout_regression — tool timeouts spike, pass rate drops slightly

Usage:
    python examples/diagnosis_demo/generate.py prompt_regression runs/prompt/ [--seed 42]
    python examples/diagnosis_demo/generate.py timeout_regression runs/timeout/ [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# --- Step generators ---

def _step(idx: int, name: str, status: str = "ok", attrs: dict | None = None) -> dict:
    return {
        "idx": idx,
        "type": "tool",
        "name": name,
        "status": status,
        "attrs": attrs or {},
        "output": {},
    }


def _generate_trajectory(
    rng: random.Random,
    strategy: str,
    task_difficulty: str,
    timeout_rate: float = 0.0,
) -> tuple[list[dict], bool]:
    """Generate a trajectory and outcome based on strategy and difficulty.

    Returns (steps, success).
    """
    steps: list[dict] = []
    idx = 0

    # Phase 1: Explore (everyone does this)
    n_reads = rng.randint(1, 3)
    for _ in range(n_reads):
        steps.append(_step(idx, "read", attrs={"file_path": f"src/module_{rng.randint(1,5)}.py"}))
        idx += 1

    if rng.random() < 0.7:
        steps.append(_step(idx, "search", attrs={"pattern": f"def {rng.choice(['foo', 'bar', 'process', 'handle'])}"}))
        idx += 1

    # Possible timeout on exploration commands (before edit)
    if timeout_rate > 0 and rng.random() < timeout_rate:
        steps.append(_step(idx, "bash", "error", attrs={"command": "python setup.py install"}))
        idx += 1
        # Often retry after timeout
        if rng.random() < 0.5:
            steps.append(_step(idx, "bash", "error", attrs={"command": "pip install -e ."}))
            idx += 1

    # Phase 2: Edit
    steps.append(_step(idx, "edit", attrs={"file_path": "src/main.py"}))
    idx += 1

    # Phase 3: Strategy-dependent — test or submit
    if strategy == "test_verify":
        # Run tests, possibly fix and re-test
        test_status = "ok" if rng.random() < 0.6 else "error"
        steps.append(_step(idx, "test", test_status, attrs={"command": "python -m pytest"}))
        idx += 1

        if test_status == "error":
            # Fix and re-test
            steps.append(_step(idx, "read"))
            idx += 1
            steps.append(_step(idx, "edit", attrs={"file_path": "src/main.py"}))
            idx += 1
            steps.append(_step(idx, "test", "ok", attrs={"command": "python -m pytest"}))
            idx += 1

    # Phase 4: Submit
    steps.append(_step(idx, "finish"))

    # Outcome depends on strategy + difficulty + model capability.
    # Probabilities tuned so that baseline (~78%) ≈ current (~77%) overall,
    # despite very different strategy distributions. This is Simpson's paradox:
    # current does better on easy tasks (model upgrade) but worse on hard tasks
    # (lost testing strategy). Net rate is nearly identical.
    if task_difficulty == "hard":
        if strategy == "test_verify":
            success = rng.random() < 0.75
        else:
            success = rng.random() < 0.48
    else:  # easy
        if strategy == "test_verify":
            success = rng.random() < 0.80
        else:
            success = rng.random() < 0.92

    return steps, success


# --- Scenario definitions ---

def _generate_prompt_regression(
    rng: random.Random,
    output_dir: Path,
    n_tasks: int = 10,
    runs_per_task: int = 10,
) -> dict:
    """Scenario: prompt change removes testing instruction.

    Baseline: 90% test-verify strategy
    Current: 30% test-verify (prompt removed instruction)
    Pass rates end up similar due to Simpson's paradox.
    """
    tasks = [f"task_{i:03d}" for i in range(n_tasks)]
    n_hard = max(1, n_tasks * 3 // 10)  # 30% hard, 70% easy for Simpson's paradox balance
    hard_tasks = set(tasks[:n_hard])
    stats = {"baseline": {"pass": 0, "total": 0}, "current": {"pass": 0, "total": 0}}

    for variant in ["baseline", "current"]:
        test_rate = 0.90 if variant == "baseline" else 0.30

        for task_id in tasks:
            difficulty = "hard" if task_id in hard_tasks else "easy"
            for run_num in range(runs_per_task):
                strategy = "test_verify" if rng.random() < test_rate else "blind_submit"
                steps, success = _generate_trajectory(rng, strategy, difficulty)

                run = {
                    "run_id": f"{variant}_{task_id}_run{run_num}",
                    "task_id": task_id,
                    "agent": "demo-agent",
                    "model": "claude-4-sonnet" if variant == "current" else "claude-3.5-sonnet",
                    "harness": "swebench",
                    "tags": {"variant": variant, "difficulty": difficulty},
                    "steps": steps,
                    "result": {"success": success},
                }

                out_path = output_dir / f"{run['run_id']}.json"
                out_path.write_text(json.dumps(run, indent=2) + "\n")

                stats[variant]["total"] += 1
                if success:
                    stats[variant]["pass"] += 1

    return stats


def _generate_timeout_regression(
    rng: random.Random,
    output_dir: Path,
    n_tasks: int = 10,
    runs_per_task: int = 10,
) -> dict:
    """Negative control: tool timeout causes regression.

    Baseline: 3% timeout rate, 90% test-verify
    Current: 28% timeout rate, 85% test-verify (testing still happens, just timeouts)
    """
    tasks = [f"task_{i:03d}" for i in range(n_tasks)]
    hard_tasks = set(tasks[:n_tasks // 2])
    stats = {"baseline": {"pass": 0, "total": 0}, "current": {"pass": 0, "total": 0}}

    for variant in ["baseline", "current"]:
        test_rate = 0.90 if variant == "baseline" else 0.85
        timeout_rate = 0.03 if variant == "baseline" else 0.45

        for task_id in tasks:
            difficulty = "hard" if task_id in hard_tasks else "easy"
            for run_num in range(runs_per_task):
                strategy = "test_verify" if rng.random() < test_rate else "blind_submit"
                steps, success = _generate_trajectory(
                    rng, strategy, difficulty, timeout_rate=timeout_rate,
                )

                # Timeouts reduce success probability
                has_timeout = any(s["status"] == "error" for s in steps)
                if has_timeout and success:
                    success = rng.random() < 0.3  # timeout usually kills the run

                run = {
                    "run_id": f"{variant}_{task_id}_run{run_num}",
                    "task_id": task_id,
                    "agent": "demo-agent",
                    "model": "claude-3.5-sonnet",
                    "harness": "swebench",
                    "tags": {"variant": variant, "difficulty": difficulty},
                    "steps": steps,
                    "result": {"success": success},
                }

                out_path = output_dir / f"{run['run_id']}.json"
                out_path.write_text(json.dumps(run, indent=2) + "\n")

                stats[variant]["total"] += 1
                if success:
                    stats[variant]["pass"] += 1

    return stats


SCENARIOS = {
    "prompt_regression": _generate_prompt_regression,
    "timeout_regression": _generate_timeout_regression,
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic regression demo data")
    parser.add_argument("scenario", choices=list(SCENARIOS.keys()), help="Scenario to generate")
    parser.add_argument("output_dir", type=Path, help="Output directory for runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--runs-per-task", type=int, default=10, help="Runs per task per variant")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    generate_fn = SCENARIOS[args.scenario]
    stats = generate_fn(rng, args.output_dir, n_tasks=args.tasks, runs_per_task=args.runs_per_task)

    for variant, s in stats.items():
        rate = s["pass"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"{variant}: {s['pass']}/{s['total']} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
