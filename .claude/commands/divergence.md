Analyze why agent runs diverge for a specific task.

Run this command to generate a structured pass/fail comparison:

```
moirai divergence $ARGUMENTS
```

This outputs a comparison document showing:
- Where a passing and failing run diverge (aligned trajectory diff)
- What each agent was thinking at the divergence point (reasoning excerpts)
- What code each agent wrote (edit diffs with old/new strings)
- What happened when they tested (test output)

Analyze the output and answer:

1. **Why did the pass run succeed?** What specific knowledge, decision, or strategy at the divergence point led to success?
2. **Why did the fail run fail?** What did it miss, misunderstand, or do differently that caused the failure? Look at the actual edit diffs — are the code changes functionally different, or is the failure from something else (eval harness criteria, test environment, etc.)?
3. **Is the failure real?** If both runs produce similar edits and both pass the same tests, the "fail" label might be a false negative from the eval harness. Call this out.
4. **What intervention would help?** Suggest a specific system prompt addition, tool restriction, or workflow change that would steer failing agents toward the successful approach. Be concrete — give the actual text to add, not abstract advice.

If no task is specified, list the available tasks with mixed outcomes first.
