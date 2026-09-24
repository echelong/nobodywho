# Decision router: bounded second opinion on routing choices

The shell command `decision ask` gives an advisory answer to ONE narrow, closed routing question. It advises; you still own permissions, safety checks and execution.

## When to call it (before acting)
- choosing between materially different execution routes or implementation approaches
- after an approach has failed (especially more than once), before choosing the recovery route; never blindly retry
- deciding whether substantial extra repository research is worth it
- choosing among several tools or skills, or whether to delegate/spawn work
- proceeding when uncertainty would materially change the next step

## When NOT to call it
Simple factual answers, deterministic calculations, routine file reads, straightforward edits, obvious commands, or any case where no answer could change your next action. Never ask it whether to use it, and never call it from inside a decision call.

If the user says "bypass decision router" or "bypass jev", do not call it for the rest of the task.

## How
```
decision ask <<'JSON'
{"question_id":"next_step",
 "state":"<compact evidence only: goal, what was tried, exact errors, constraints>",
 "question":"<one narrow closed question>",
 "choices":{"<descriptive_id>":"<one-line action>","<descriptive_id>":"<one-line action>","<descriptive_id>":"<one-line action>"},
 "allow_abstain":true,"risk":"low|medium|high"}
JSON
```
- 2-6 choices with descriptive snake_case ids (e.g. `inspect_timezone_handling`, not `option_a`), each a concrete next action; include the obvious one.
- State: only the evidence needed. No secrets, keys, tokens, env dumps or whole files.
- Read `follow` in the JSON output. If it is set, take that route unless it conflicts with the user's instructions or safety. If it is null (off, bypass, abstain, error, disagreement), use your own judgement. In shadow mode only `decision` counts; `shadow` is informational.
- State the chosen route in one line, then continue the ORIGINAL task.
