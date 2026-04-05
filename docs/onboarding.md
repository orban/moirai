# Onboarding: moirai + eval harness

This doc gets you from zero to running moirai analysis on your own agent traces. It covers the trace format, the eval harness (for generating traces), and how to plug in your own agent infrastructure.

## 1. Quick start with moirai

### Install

```bash
git clone <moirai-repo-url>
cd moirai
pip install -e .
```

Requires Python 3.11+. Dependencies: typer, rich, plotly, scipy, numpy.

### Run it on example data

moirai ships with synthetic traces in `examples/`. Try these commands to see what it does:

```bash
# Aggregate stats: pass rate, step counts, top signatures
moirai summary examples/synthetic_helpdesk/

# Find behavioral patterns that predict success or failure
moirai patterns examples/synthetic_helpdesk/

# Per-task divergence analysis (needs 3+ runs per task with mixed outcomes)
moirai branch examples/swe_smith/

# Inspect a single run
moirai trace examples/synthetic_helpdesk/run_001.json
```

For real data (100+ runs of the same tasks), things get interesting. The `examples/swe_smith/` directory has ~80 runs from SWE-smith that show actual divergence patterns.

### The JSON trace format

Each trace is one JSON file, one file per run. Here's the minimal schema:

```json
{
  "run_id": "my-run-001",
  "task_id": "fix-login-redirect",
  "steps": [
    {
      "idx": 0,
      "type": "tool",
      "name": "read",
      "status": "ok",
      "attrs": {"file_path": "src/auth.py"},
      "input": {},
      "output": {},
      "metrics": {"tokens_in": 500, "tokens_out": 200}
    },
    {
      "idx": 1,
      "type": "tool",
      "name": "edit",
      "status": "ok",
      "attrs": {"file_path": "src/auth.py"}
    },
    {
      "idx": 2,
      "type": "tool",
      "name": "test",
      "status": "ok",
      "attrs": {"command": "pytest tests/test_auth.py"}
    }
  ],
  "result": {
    "success": true,
    "score": 1.0,
    "label": "pass"
  }
}
```

**Required fields:**
- `run_id` -- unique identifier for this run
- `task_id` -- which task this run attempted (moirai groups by task_id for alignment)
- `steps` -- ordered list of steps the agent took
- `steps[].idx` -- integer index (0-based)
- `steps[].type` -- one of: `llm`, `tool`, `system`, `memory`, `compaction`, `judge`, `error`, `handoff`
- `steps[].name` -- what the step did: `read`, `edit`, `write`, `search`, `bash`, `test`, `subagent`, `plan`, etc.
- `result.success` -- boolean, or null if unknown

**Optional but useful:**
- `task_family` -- groups related tasks (e.g., repo name). Used by `--stratify` and `--task-family` filters
- `harness` -- which condition/variant produced this run (e.g., "none", "flat_llm", "intent_layer")
- `model` -- model name, for filtering
- `tags` -- arbitrary key/value metadata (cost, tokens, wall clock, etc.)
- `steps[].attrs` -- structured metadata that moirai uses for enriched labels. Key attrs:
  - `file_path` -- for read/edit/write steps, moirai infers file type (source, test, config)
  - `command` -- for bash steps, moirai detects test runs vs. other commands
  - `pattern` -- for search steps, moirai distinguishes glob vs. grep vs. specific
- `steps[].output.reasoning` -- the agent's thinking/reasoning text before this step. If present, `moirai explain` can do content-aware analysis
- `steps[].output.result` -- the tool's output text
- `steps[].metrics` -- `tokens_in`, `tokens_out`, `latency_ms`
- `result.score` -- float 0-1 for graded outcomes
- `result.error_type` -- string classifying the failure mode

### Step type aliases

moirai normalizes these aliases automatically:
- `assistant` / `completion` → `llm`
- `function_call` / `retrieval` → `tool`
- `compress` → `compaction`
- `critic` → `judge`

### Converting from other formats

moirai includes converters in `scripts/`:

```bash
# From eval-harness results
python scripts/convert_eval_harness.py /path/to/eval-harness/ output_runs/

# From SWE-smith
python scripts/convert_swe_smith.py /path/to/swe-smith-runs/ output_runs/

# From OpenHands
python scripts/convert_openhands.py /path/to/openhands-output/ output_runs/

# From SWE-agent
python scripts/convert_swe_agent.py /path/to/swe-agent-output/ output_runs/

# From CoderForge
python scripts/convert_coderforge.py /path/to/coderforge-output/ output_runs/

# From SWE-bench experiment submissions
python scripts/convert_swebench_experiments.py /path/to/submissions/ output_runs/
```

These converters show the mapping between each format and moirai's schema. If you're writing your own converter, `convert_eval_harness.py` is the best reference since it handles the exact `claude --output-format stream-json` output you'll likely be producing.


## 2. The eval harness

The eval harness is an A/B testing framework for comparing agent skill configurations. It runs Claude Code on bug fix tasks inside Docker containers and measures pass rates, token usage, and tool call patterns.

### Install

```bash
cd eval-harness
pip install -e ".[dev]"
```

Prerequisites: Docker running, `claude` CLI installed, Python 3.11+.

### How task files work

A task YAML file describes a repo and a set of bug fix tasks extracted from its git history. Example from `tasks/fastmcp-focused.yaml`:

```yaml
repo:
  url: https://github.com/jlowin/fastmcp
  default_branch: main
  docker:
    image: python:3.11-slim
    setup:
    - pip install uv && uv sync --frozen --all-groups
    test_command: uv run --frozen pytest
  strip_extra:
  - .claude/
  - .cursor/

tasks:
- id: merge-pull-request-3195-from-jlowinfixstateless
  category: simple_fix
  difficulty: easy
  pre_fix_commit: 453dcbe808fc942a55b04c49a0e9478249fb2655
  fix_commit: b4eb1adbeb80f12e8d85c7a8b32f51e39cbc66fa
  prompt_source: failing_test
  test_file: tests/client/test_streamable_http.py
```

Each task is a commit that fixed a bug. The harness checks out `pre_fix_commit` (the broken state), gives Claude a prompt (either from the failing test output, the commit message, or a GitHub issue), and checks if the fix makes the tests pass.

**Generating task files from your own repos:**

```bash
eval-harness scan \
  --repo https://github.com/your-org/your-repo \
  --output tasks/your-repo.yaml \
  --limit 20 \
  --docker-image python:3.11-slim \
  --setup "pip install -e ." \
  --test-command "pytest" \
  --branch main
```

This scans git history for bug fix commits. Review the output YAML and remove tasks that are too large, trivial, or depend on external services.

### How conditions work

Each task runs under one or more "conditions" that control what context the agent gets:

| Condition | What it does |
|---|---|
| `none` | Bare Claude Code, no scaffolding. The baseline. |
| `flat_llm` | Claude first generates a CLAUDE.md overview of the codebase (AGENTbench-style), then fixes the bug with that context. |
| `intent_layer` | Claude generates a hierarchical Intent Layer (CLAUDE.md + AGENTS.md files), then fixes the bug using that richer context. |
| `human` | Uses a pre-existing human-written CLAUDE.md. |

The conditions control what preamble gets prepended to the fix prompt. For `none`, the agent just gets "Fix the following bug: ...". For `flat_llm` and `intent_layer`, the agent first generates context files, then gets a preamble telling it to read those files before attempting the fix.

### Running evals

```bash
# Basic run: all conditions, 1 rep per (task, condition) pair
eval-harness run --tasks tasks/your-repo.yaml --parallel 4

# Specific conditions only
eval-harness run --tasks tasks/your-repo.yaml --condition none --condition flat_llm

# Multiple repetitions for statistical power (moirai needs repeated runs)
eval-harness run --tasks tasks/your-repo.yaml \
  --parallel 8 \
  --repetitions 5 \
  --condition none

# Dry run to see what would execute
eval-harness run --tasks tasks/your-repo.yaml --dry-run

# Resume a previous run (skip already-completed tasks)
eval-harness run --tasks tasks/your-repo.yaml --resume results/2026-01-21-143200.json

# Keep workspaces for debugging
eval-harness run --tasks tasks/your-repo.yaml --keep-workspaces
```

For moirai analysis to work well, you want **5+ repetitions** of each (task, condition) pair. This gives enough runs per task to detect behavioral divergence.

### Where results go

```
results/
  2026-01-21-143200.json   # structured results (pass/fail, tokens, timing)
  2026-01-21-143200.md     # human-readable summary
  trials/
    taskid-none-r0.json    # per-trial results
    taskid-flat_llm-r0.json
    ...
logs/
  taskid-none-r0.log       # tool-by-tool execution log ([tool] Read src/foo.py)
  ...
```

The `trials/` directory + `logs/` directory is what the moirai converter reads. The log files contain the tool call sequence that becomes the moirai trace.


## 3. Generating your own traces

You have a Docker-based agent setup with an externalized Python state machine invoking Claude Code. Here's how to capture traces moirai can consume.

### Option A: Capture Claude Code stream-json output directly

If you're invoking `claude` CLI inside Docker, use `--output-format stream-json`. Each line is a JSON event:

```bash
claude --print \
  --output-format stream-json \
  --max-turns 50 \
  --dangerously-skip-permissions \
  "Fix the bug described in issue #42" \
  > /output/stream.jsonl
```

The stream-json output looks like this (one JSON object per line):

```jsonl
{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"Let me look at..."}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","id":"tu_01","input":{"file_path":"src/auth.py"}}]}}
{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_01","content":"...file contents..."}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Edit","id":"tu_02","input":{"file_path":"src/auth.py","old_string":"...","new_string":"..."}}]}}
{"type":"result","num_turns":5,"total_cost_usd":0.15,"usage":{"input_tokens":5000,"output_tokens":2000}}
```

### Option B: Write a converter from your state machine logs

If your agent infrastructure already logs actions in its own format, write a converter. Here's a minimal template:

```python
#!/usr/bin/env python3
"""Convert custom agent logs to moirai trace format."""
import json
import sys
from pathlib import Path


def convert_run(log_path: Path, task_id: str, run_id: str, success: bool) -> dict:
    """Convert one agent run log into a moirai trace."""
    steps = []
    idx = 0

    for line in log_path.read_text().splitlines():
        # Adapt this to your log format
        event = json.loads(line)

        step = {
            "idx": idx,
            "type": classify_type(event),     # "tool", "llm", "system", etc.
            "name": classify_name(event),     # "read", "edit", "test", "bash", etc.
            "status": "ok",
            "attrs": extract_attrs(event),    # {"file_path": "...", "command": "..."}
            "output": {},
        }

        # If you have reasoning/thinking text, include it
        if event.get("thinking"):
            step["output"]["reasoning"] = event["thinking"]

        # If you have tool output, include it
        if event.get("result"):
            step["output"]["result"] = str(event["result"])[:2000]

        steps.append(step)
        idx += 1

    return {
        "run_id": run_id,
        "task_id": task_id,
        "task_family": task_id.split("-")[0],  # or however you group tasks
        "agent": "my-agent",
        "model": "claude-sonnet-4-20250514",
        "harness": "my-harness-v1",
        "tags": {},
        "steps": steps,
        "result": {
            "success": success,
            "score": 1.0 if success else 0.0,
        },
    }


def classify_type(event: dict) -> str:
    """Map your event types to moirai step types."""
    kind = event.get("type", "")
    if kind in ("read", "edit", "write", "search", "bash", "test"):
        return "tool"
    if kind in ("think", "plan", "reason"):
        return "llm"
    if kind == "subagent":
        return "llm"
    return "system"


def classify_name(event: dict) -> str:
    """Map your event names to moirai step names.

    The name is what moirai uses for alignment and pattern mining.
    More granular names = better analysis. Key names moirai understands:
      read, edit, write, search, bash, test, subagent, plan
    """
    action = event.get("action", "")
    # Detect test runs
    if "pytest" in action or "npm test" in action or "cargo test" in action:
        return "test"
    if action.startswith("read"):
        return "read"
    if action.startswith("edit") or action.startswith("write"):
        return "edit"
    if action.startswith("grep") or action.startswith("find"):
        return "search"
    return "bash"


def extract_attrs(event: dict) -> dict:
    """Extract structured attributes moirai uses for enriched labels.

    file_path → moirai infers: read(source), read(test_file), read(config)
    command   → moirai infers: bash(python), bash(git), test
    pattern   → moirai infers: search(glob), search(specific)
    """
    attrs = {}
    if event.get("file"):
        attrs["file_path"] = event["file"]
    if event.get("command"):
        attrs["command"] = event["command"]
    if event.get("pattern"):
        attrs["pattern"] = event["pattern"]
    return attrs


# --- main ---
if __name__ == "__main__":
    log_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    for log_file in sorted(log_dir.glob("*.jsonl")):
        # You'll need to extract task_id, success, etc. from your logs or a manifest
        task_id = log_file.stem.rsplit("-", 1)[0]
        run_id = log_file.stem
        success = True  # replace with actual outcome

        run = convert_run(log_file, task_id, run_id, success)
        out = output_dir / f"{run_id}.json"
        out.write_text(json.dumps(run, indent=2))

    print(f"Converted {len(list(log_dir.glob('*.jsonl')))} runs to {output_dir}")
```

### Converting stream-json to moirai format

If you're capturing `stream-json` output from Claude Code, here's how to convert it:

```python
#!/usr/bin/env python3
"""Convert Claude Code stream-json JSONL to moirai trace format."""
import json
from pathlib import Path


def stream_json_to_moirai(
    jsonl_path: Path,
    task_id: str,
    run_id: str,
    success: bool,
) -> dict:
    steps = []
    idx = 0
    pending_reasoning = None

    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        etype = event.get("type", "")

        if etype == "assistant":
            content = event.get("message", {}).get("content", [])
            for block in content:
                bt = block.get("type")

                if bt == "thinking":
                    pending_reasoning = block.get("thinking", "")

                elif bt == "tool_use":
                    name_raw = block.get("name", "unknown")
                    inp = block.get("input", {})

                    # Map Claude Code tool names to moirai names
                    name, stype, attrs = _map_tool(name_raw, inp)

                    step = {
                        "idx": idx,
                        "type": stype,
                        "name": name,
                        "status": "ok",
                        "attrs": attrs,
                        "output": {},
                    }
                    if pending_reasoning:
                        step["output"]["reasoning"] = pending_reasoning
                        pending_reasoning = None

                    steps.append(step)
                    idx += 1

        # Match tool results back to their steps
        elif etype == "user":
            content = event.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    result_text = block.get("content", "")
                    if isinstance(result_text, list):
                        result_text = "\n".join(
                            p.get("text", "") for p in result_text if isinstance(p, dict)
                        )
                    # Find the matching step and attach the result
                    # (simplified: attach to last step)
                    if steps:
                        steps[-1]["output"]["result"] = str(result_text)[:2000]

    return {
        "run_id": run_id,
        "task_id": task_id,
        "steps": steps,
        "result": {"success": success},
    }


def _map_tool(name: str, inp: dict) -> tuple[str, str, dict]:
    """Map Claude Code tool name + input to (moirai_name, step_type, attrs)."""
    attrs = {}

    if name == "Read":
        attrs["file_path"] = inp.get("file_path", "")
        return "read", "tool", attrs

    if name in ("Edit", "Write"):
        attrs["file_path"] = inp.get("file_path", "")
        return "edit", "tool", attrs

    if name == "Bash":
        cmd = inp.get("command", "")
        attrs["command"] = cmd[:200]
        if any(kw in cmd.lower() for kw in ["pytest", "test", "npm test", "cargo test"]):
            return "test", "tool", attrs
        return "bash", "tool", attrs

    if name in ("Glob", "Grep"):
        attrs["pattern"] = inp.get("pattern", "")
        return "search", "tool", attrs

    if name == "Agent":
        return "subagent", "llm", attrs

    if name == "TodoWrite":
        return "plan", "system", attrs

    return name.lower(), "tool", attrs
```

### What matters for good analysis

Three things make traces useful:

1. **Same task_id for repeated runs.** moirai aligns runs of the same task. If you run task X five times, all five traces need `"task_id": "X"`. Without this, there's nothing to align.

2. **Mixed outcomes.** You need both passing and failing runs of the same task. If a task always passes (or always fails), there's no divergence to find. Run enough repetitions (5-10) that you get variance.

3. **Granular step names with attrs.** `"name": "tool"` is nearly useless. `"name": "read"` with `"attrs": {"file_path": "src/auth.py"}` is good -- moirai will enrich that to `read(source)` and use the file type in pattern mining. The more semantic your step names are, the more moirai can find.


## 4. Running moirai analysis on your traces

Assume you have a directory of JSON trace files at `runs/`.

### Summary: what happened

```bash
moirai summary runs/
```

Shows run count, pass rate, average/median steps, top trajectory signatures, and error type distribution. Quick sanity check that your traces loaded correctly.

```bash
# Filter to a specific condition or model
moirai summary runs/ --harness none
moirai summary runs/ --model claude-sonnet-4-20250514
moirai summary runs/ --task-family express
```

### Branch: where trajectories diverge

```bash
moirai branch runs/
```

Groups runs by task_id, aligns repeated runs using Needleman-Wunsch, and finds positions where different choices correlate with different outcomes. Shows the top divergence points per task with their q-values (BH-adjusted p-values).

```bash
# Generate an HTML report with heatmap + dendrogram
moirai branch runs/ --html report.html

# Interactive heatmap viewer (self-contained HTML)
moirai branch runs/ --viewer viewer.html

# With LLM-assisted analysis of divergence points
moirai branch runs/ --html report.html --analyze
```

The HTML report is the most useful output. Each task with mixed outcomes gets a panel showing:
- A heatmap of aligned trajectories (rows = runs, columns = aligned steps)
- A dendrogram clustering similar runs
- Numbered badges at divergence points
- A side-by-side comparison of a passing and failing run at the divergence

### Patterns: what predicts success

```bash
moirai patterns runs/
```

Mines 3-5 step n-grams from all runs and tests which patterns discriminate between success and failure using Fisher's exact test with BH correction.

```bash
# Use --stratify to prevent cross-task-family confounds
# (Simpson's paradox is real -- patterns that look significant across all tasks
#  may just reflect that some task families are easier than others)
moirai patterns runs/ --stratify task_family

# Include gapped patterns (ordered subsequences with flexible gaps)
moirai patterns runs/ --stratify task_family --gapped

# Empirical FDR validation
moirai patterns runs/ --stratify task_family --permutation-test 1000
```

`--stratify task_family` is almost always what you want. Without it, you'll find patterns that are really just "tasks from repo X are easier" in disguise.

### Export: extract DPO preference pairs

```bash
moirai export runs/ --format dpo --output pairs.jsonl
```

For each task with mixed outcomes, aligns pass/fail trajectory pairs and extracts (prompt, chosen, rejected) triples at divergence points. Each line of the output JSONL is one preference pair:

```json
{
  "prompt": "... shared context steps leading to the fork ...",
  "chosen": "... step content from the passing run ...",
  "rejected": "... step content from the failing run ...",
  "task_id": "fix-login-redirect",
  "divergence_column": 15,
  "preferred_success_rate": 0.85,
  "dispreferred_success_rate": 0.10,
  "p_value": 0.003
}
```

Useful for fine-tuning or RLHF if you have enough traces with reasoning content.

```bash
# Tune extraction parameters
moirai export runs/ --format dpo \
  --max-per-task 25 \
  --min-steps 3 \
  --min-content 50 \
  --max-position 0.8
```

### Other useful commands

```bash
# Compare two conditions side by side
moirai diff runs/ --a harness=none --b harness=intent_layer

# Explain a specific run in cluster context
moirai explain runs/ --run my-run-001

# Cross-run content-aware analysis (needs claude or codex CLI)
moirai explain runs/ --mode claude --task fix-login-redirect

# Structured pass/fail comparison for piping to an LLM
moirai divergence runs/ --task fix-login-redirect | pbcopy

# Validate your trace files before analysis
moirai validate runs/

# Combined HTML report (summary + patterns + branch in one page)
moirai report runs/ --html full-report.html

# Extract behavioral feature shifts between variants
moirai evidence runs/ --baseline harness=none --current harness=intent_layer

# Diagnose regression causes with a candidate cause file
moirai diagnose runs/ --baseline harness=none --current harness=intent_layer --causes causes.json
```

### Typical workflow

1. Run your agent on a set of tasks with 5-10 repetitions each
2. Convert traces to moirai format (one JSON file per run)
3. `moirai validate runs/` -- make sure everything loads
4. `moirai summary runs/` -- check pass rates and step counts
5. `moirai branch runs/ --html report.html` -- find where behavior splits
6. `moirai patterns runs/ --stratify task_family` -- find predictive patterns
7. `moirai explain runs/ --mode claude` -- get LLM-assisted explanations of divergence

The branch HTML report is usually where the actionable insights are. Look for divergence points where one choice leads to 80%+ pass rate and the other to 0-20%. Those are the decision points worth investigating.
