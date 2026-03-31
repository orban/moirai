Analyze eval results to find actionable failure patterns.

First, get an overview of the eval results:

```
moirai summary $ARGUMENTS
```

Then find tasks with mixed outcomes (some pass, some fail — these are where the signal is):

```
moirai branch $ARGUMENTS
```

For each task with mixed outcomes, generate a detailed divergence analysis:

```
moirai divergence $ARGUMENTS --task <TASK_ID>
```

For each divergence analysis:
1. Identify whether the divergence is meaningful (different strategies) or noise (same strategy, different eval outcome)
2. If meaningful: what specific agent behavior predicts success vs failure?
3. Synthesize findings across all tasks into actionable recommendations

Present findings as:
- **Pattern summary**: what recurring behaviors distinguish pass from fail runs?
- **Top interventions**: ranked list of specific changes (system prompt additions, tool restrictions) that would improve the pass rate, with expected impact
- **False negatives**: tasks where the fail classification appears incorrect (agent produced correct code but eval harness rejected it)
