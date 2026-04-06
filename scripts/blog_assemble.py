"""Assemble the Hugo blog post with inline SVGs.

Narrative structure (each concept builds on the last):
  1. Hook: 50/50 pass/fail — why?
  2. Step classification: each step → trajectory bars
  3. Alignment: NW to compare runs → divergence detection
  4. Scale up: divergence clustering → failure modes
  5. Validate: within-task experiments → feature table
  6. Re-examine: use discovered features → test timing alignment
  7. Population: beeswarm → mechanism: flow diagram
  8. Next steps: intervention study needed
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "blog_output")
BLOG_DIR = os.path.expanduser("~/dev/web/ryanorban.com/content/posts")


def read_svg(name: str) -> str:
    with open(os.path.join(OUTPUT_DIR, f"{name}.svg")) as f:
        return f.read()


# ── Prose sections ─────────────────────────────────────────────────

FRONTMATTER = """\
---
title: "What Stochastic Variation Reveals About AI Agents"
date: 2026-04-04
description: "Same agent, same task, different outcomes. Here's what the variation tells you about why agents fail — and what to do about it."
---
"""

INTRO = """\
This is a sequel to [Stop Testing AI Agents Like Deterministic Code](/posts/stop-testing-agents-like-deterministic-code/). That post argued you should treat agents as stochastic processes (same inputs, probabilistic outputs). This one shows what you find when you do.

The setup: one agent ([OpenHands](https://github.com/All-Hands-AI/OpenHands) running Qwen3-Coder-480B) attempts 1,096 software engineering tasks from [SWE-rebench](https://github.com/nebius/swe-rebench), each 4 to 33 times (median 11). Same code, same prompt, same environment. 12,854 runs total. Every one of these tasks has mixed outcomes — the same agent sometimes succeeds and sometimes doesn't.

---

## Same task, different outcomes

Pick one task. [vyper #4385](https://github.com/vyperlang/vyper/issues/4385). The agent tries it 10 times. Five pass, five fail.

What does each attempt look like? We classify each step the agent takes — reading files, searching code, editing, running tests, reasoning — and get a *trajectory*: the sequence of actions it took to attempt the fix.

"""

AFTER_TRAJECTORIES = """\

Each bar is one run. Five pass (green border), five fail (red). They all start similarly: read the repo, run setup commands, explore the codebase. Then they diverge. But comparing the raw trajectories doesn't reveal much — the differences are subtle and the sequences are different lengths.

We need a way to compare them.

---

## Aligning the trajectories

We borrow a technique from bioinformatics: [Needleman-Wunsch alignment](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm), originally designed to align DNA sequences. It snaps matching steps into columns. Where runs take different paths, gaps appear.

With the runs aligned, we can test each column statistically: does having a step here (vs a gap) predict pass or fail? [Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) on each column, ranked by significance:

"""

AFTER_DIVERGENCE = """\

The orange strip shows the p-value per column — brighter means more significant. The top-ranked divergence points show where pass and fail runs make different choices.

Pause on that. These aren't different agents. They aren't different prompts or different models. This is the same agent, the same code, the same task, the same environment — run ten times. The initial divergences come from stochastic sampling: at some early step, the LLM generates a different token, and the trajectory forks. Everything downstream is conditioned on that fork — the agent reads different files, builds different context, makes different decisions. One random choice cascades into a different strategy.

On a single task with n=10, individual divergence points are noisy and task-specific. But the technique scales.

---

## Scaling up: recurring failure modes

Running this alignment across all 1,096 mixed-outcome tasks and clustering the divergence points by content (TF-IDF on step descriptions) reveals recurring patterns:

```bash
moirai divergences examples/swe_rebench --output divergences.json
```

"""

AFTER_FAILURE_MODES = """\

The largest category is search strategy divergence (31%): at the same point in the trajectory, pass and fail runs choose different search approaches. "Wasted orientation" (18%) means the agent kept re-reading files it had already seen — spinning, not exploring. "Wrong file targeted" (12%) means it committed to editing the wrong source file and never recovered.

These patterns suggest measurable behavioral features. If agents that search differently succeed more often, can we measure *how* they search? If timing matters at divergence points, can we measure *when* key actions happen?

---

## Validating the patterns

We formalize the divergence observations into computable features — when testing happens, how edits are distributed, what the reasoning language looks like — and validate each with a within-task natural experiment: split runs at the median feature value, compare pass rates. This controls for task identity because the comparison is always within the same task.

```bash
moirai features examples/swe_rebench --min-runs 4 --output results.json
```

Five features survive with significant effects (not every task has data for every feature, so the exact count varies slightly):

"""

AFTER_FEATURES = """\

Uncertainty and struggle markers are the strongest signal: runs where the agent hedges or backtracks ("maybe," "might," "let me try," "another approach") fail 7 percentage points more often. This likely reflects a confused agent rather than hedging language causing failure — but it's detectable either way. Trajectory shape is second: runs that follow a clean explore-then-modify-then-verify arc pass 6.6pp more often. Test timing is close behind at +6.5pp. Hypothesis formation predicts failure — the agent that says "I think the issue might be..." performs worse than the one that just searches until it finds the answer.

These are within-task effects from stochastic variation. An early random choice — which file to read first, which search to run — cascades into different downstream behaviors. The initial divergence is stochastic; the features capture where the cascade leads.

---

## Re-examining with the discovered features

Now we can go back to our vyper task with the features we've discovered and validated. The `--feature` flag connects the discovery tool to the drill-down:

```bash
moirai branch examples/swe_rebench --task "vyperlang__vyper-4385" \\
    --feature test_position_centroid
```

Here are the same 10 runs, now sorted by test timing (the third-strongest feature) and annotated with each run's test centroid:

"""

AFTER_ANNOTATED_ALIGNMENT = """\

The border gradient encodes test centroid — orange for early testing (top), teal for late testing (bottom). The dot strip below shows the same data as points on a number line. The separation is stark: every early tester fails, every late tester passes.

We selected this task because the effect is unusually clean. Does it hold up across tasks?

"""

AFTER_BEESWARM = """\

Each dot is one task. For each, we split the runs at the median test centroid and compared pass rates. Dots right of zero mean later testing predicted success on that task. The distribution leans right: 57% of tasks show this pattern, 43% show the opposite (sign test, p < 0.001). The per-task effect is weak — the significance comes from consistency across a thousand tasks, not from any single task.

On our vyper task, the mechanism is visible:

"""

AFTER_FLOW = """\

On this task, the mechanism is visible: runs that happen to test early get results they can't interpret yet, enter bash loops, and never converge. Runs that search first and test after their changes enter a productive edit-test cycle. Whether this is causal (early testing derails the agent) or consequential (confused agents happen to test early) is an open question — but both interpretations are detectable in real time.

---

## From observation to training signal

Every agent eval produces trajectories. Most of the time, those trajectories get a pass/fail label and get thrown away. But when the same agent passes and fails on the same task, the difference between those trajectories is a training signal — a specific moment where a different choice would have led to a different outcome.

The features are predictive, not causal. But they don't have to be causal to be useful — they just have to point at the right moments.

At every divergence point, moirai extracts a preference pair: the context both runs shared, the step the pass run took (chosen), and the step the fail run took (rejected). That's the input format for [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

```bash
moirai export --format dpo examples/swe_rebench --output pairs.jsonl
# → 11,006 preference pairs from 1,096 tasks
```

Each pair captures a specific moment where stochastic variation determined the outcome. The pass run searched deeper, or tested later, or avoided hedging — and that choice cascaded into success. The fail run did the opposite.

The behavioral features serve a second purpose: process reward signals. Instead of the sparse pass/fail reward at the end of a trajectory, the features provide dense, step-level signal. An agent with a test centroid of 0.3 at step 20 is on a path that predicts failure — that's a reward shaping signal you can act on before the trajectory finishes.

The pipeline is:
1. Run your agent many times (stochastic variation generates diverse trajectories for free)
2. `moirai features` identifies which behaviors predict success
3. `moirai divergences` finds the decision points where outcomes split
4. `moirai export --format dpo` extracts preference pairs at those points
5. Fine-tune on the pairs, or build a process reward model from the features

We haven't closed the loop yet — the DPO fine-tuning and reward model training are next. But the extraction pipeline is working, the data is there (11,006 pairs with reasoning content on both sides), and the behavioral features provide the process-level signal that outcome-only rewards miss.

---

## Reproduce this

The methodology and code are open source: [github.com/orban/moirai](https://github.com/orban/moirai). The data is from [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) on HuggingFace (CC-BY-4.0).

```bash
pip install -e .
python scripts/convert_swe_rebench.py /path/to/downloaded/trajectories examples/swe_rebench
moirai features examples/swe_rebench --min-runs 4 --output results.json
moirai export --format dpo examples/swe_rebench --output pairs.jsonl
```

---

## Scope and corrections

Everything in this post comes from one agent on one benchmark. The patterns are real for OpenHands + Qwen3-Coder on [SWE-rebench](https://github.com/nebius/swe-rebench) tasks, but they may not generalize to other agents or task types.

We caught and corrected a significant bug during this analysis. Our initial test detection had zero true positives — 3,637 pytest invocations were misclassified because they started with a directory change prefix. Fixing that collapsed our flagship metric from +39pp to +14.9pp and changed which features survived validation. If you're doing behavioral analysis on agent traces, verify your feature extraction before trusting your features.
"""


# ── Assembly helpers ───────────────────────────────────────────────

def _add_a11y(svg: str, role_label: str, title: str) -> str:
    svg = svg.replace("<svg ", f'<svg role="img" aria-label="{title}" ', 1)
    first_close = svg.index(">") + 1
    svg = svg[:first_close] + f"\n<title>{title}</title>" + svg[first_close:]
    if "max-width" not in svg:
        svg = svg.replace('style="', 'style="max-width:700px;width:100%;height:auto;', 1)
        if 'style="' not in svg[:200]:
            svg = svg.replace("<svg ", '<svg style="max-width:700px;width:100%;height:auto" ', 1)
    return svg


def main():
    trajectories = _add_a11y(
        read_svg("panel1_hook"),
        "Ten trajectory bars showing five passing and five failing runs",
        "Same task, same agent, ten runs: five pass, five fail",
    )
    divergence = _add_a11y(
        read_svg("panel_divergence"),
        "Aligned trajectories with Fisher divergence strip and ranked divergence points",
        "Needleman-Wunsch alignment with per-column significance testing",
    )
    failure_modes = _add_a11y(
        read_svg("panel3_scale"),
        "Horizontal bar chart of failure modes from divergence clustering",
        "Failure modes at scale from moirai divergences output",
    )
    features = _add_a11y(
        read_svg("panel4_predictors"),
        "Diverging bar chart of five behavioral features predicting success or failure",
        "Behavioral features ranked by within-task effect size",
    )
    annotated = _add_a11y(
        read_svg("panel2a_alignment"),
        "Aligned trajectories sorted by test timing with centroid annotations",
        "10 runs sorted by test centroid with gradient borders and dot strip",
    )
    beeswarm = _add_a11y(
        read_svg("panel2c_beeswarm"),
        "Beeswarm plot of per-task test timing deltas across 1032 tasks",
        "Population evidence: 57% of tasks show later testing predicts success",
    )
    flow = _add_a11y(
        read_svg("panel2b_flow"),
        "Decision flow showing two strategies from stochastic variation",
        "Two strategies: test early and get stuck vs explore first and test after editing",
    )

    post = (
        FRONTMATTER
        + INTRO
        + trajectories + "\n"
        + AFTER_TRAJECTORIES
        + divergence + "\n"
        + AFTER_DIVERGENCE
        + failure_modes + "\n"
        + AFTER_FAILURE_MODES
        + features + "\n"
        + AFTER_FEATURES
        + annotated + "\n"
        + AFTER_ANNOTATED_ALIGNMENT
        + beeswarm + "\n"
        + AFTER_BEESWARM
        + flow + "\n"
        + AFTER_FLOW
    )

    out_path = os.path.join(BLOG_DIR, "what-stochastic-variation-reveals.md")
    with open(out_path, "w") as f:
        f.write(post)

    print(f"Written to {out_path}")
    print(f"Post size: {len(post):,} bytes")
    lines = post.split("\n")
    print(f"Lines: {len(lines)}")
    prose_words = sum(
        len(line.split())
        for line in lines
        if not line.strip().startswith("<")
    )
    print(f"Prose words: ~{prose_words}")


if __name__ == "__main__":
    main()
