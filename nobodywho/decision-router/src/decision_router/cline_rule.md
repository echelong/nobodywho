# Decision router: bounded second opinion at real forks

The shell command `decision ask` gives an advisory answer to ONE narrow, closed question about what to do next. It advises; you still own permissions, safety checks and execution.

**Checkpoint before each action:** is this (a) a retry or recovery after a failure, including a failure the user reported, (b) a pick between materially different approaches, tools or delegation routes, or (c) the start of a large investigation? If yes, and you have no answer yet for this decision point, run `decision ask` first.

## Call it first, before acting, when any of these is true
1. You are about to choose between materially different implementation paths or execution routes.
2. Something just failed (command, test, build, tool) and you are about to retry it or pick a recovery route. After the same approach has failed twice (including failures the user reports), call it before running that command again in any form, even just to reproduce the error; if you want the full error first, offer `reproduce_failure` as one of the choices.
3. You are about to start substantial extra investigation (broad searches, reading many files, probing the system) and it is unclear whether it is worth it or which direction to take.
4. You are about to choose among genuinely different tools, skills or delegation routes (for example: do it inline, script it, or spawn subagents).
5. Uncertainty would materially change what you do next.

## Do not call it for
Simple arithmetic, simple factual questions, routine file reads, deterministic commands, obvious edits, or any choice where no answer could change your next action. An edit is only obvious when there is one reasonable way to make it; a failure with several plausible fixes is a recovery decision (item 2). Ask once per decision point. Never ask it whether to use it, and never call it from inside a decision call.

If the user says "bypass decision router" or "bypass jev", do not call it for the rest of the task.

## How
```
decision ask <<'JSON'
{"question_id":"next_step",
 "state":"<compact evidence only: goal, what was tried, exact error lines, constraints>",
 "question":"<one narrow closed question>",
 "choices":{"<descriptive_id>":"<one-line action>","<descriptive_id>":"<one-line action>","<descriptive_id>":"<one-line action>"},
 "allow_abstain":true,"risk":"low|medium|high"}
JSON
```
- 2-6 choices with descriptive snake_case ids (e.g. `inspect_timezone_handling`, not `option_a`), each a concrete next action; include the obvious one (such as retrying unchanged).
- State: only the evidence needed. No secrets, keys, tokens, env dumps or whole files.
- Read `follow` in the JSON output. If it is set, take that route unless it conflicts with the user's instructions or safety. If it is null (off, bypass, abstain, error, disagreement), use your own judgement. In shadow mode only `decision` counts; `shadow` is informational.
- State the chosen route in one line, then continue the ORIGINAL task.
