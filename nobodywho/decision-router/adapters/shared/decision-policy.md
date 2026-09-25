# Shared local decision support

Before a meaningful bounded choice whose answer changes the next action, run
`decision ask --caller "${DECISION_ROUTER_CALLER:-@CALLER@}"` with compact evidence
and 2–6 concrete choices. For example:

`decision ask --caller "${DECISION_ROUTER_CALLER:-@CALLER@}" --json '{"state":"brief facts","question":"Which route?","choices":{"first_route":"First action","second_route":"Second action"},"allow_abstain":true}'`

Use snake_case choice keys. Follow the JSON `follow` field when safe; if null,
use your judgement. The shared router chooses local tiers.

Skip obvious reads and edits, deterministic commands, arithmetic, and routine
actions. Never ask the router whether to call itself. If the user says
"bypass decision router" or "bypass jev", skip it for that task.

Use the existing shared output-pruning adapter for long captured command output
where available. Do not run a second pruner or contact TypeSafe directly.
