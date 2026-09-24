#!/usr/bin/bash
# Codex 0.156.1 PreToolUse Bash hook. jq quotes the original command as one
# shell argument; the shared runner executes it in Codex's existing shell.
runner=/home/rio/Projects/nobodywho/nobodywho/decision-router/adapters/shared/decision-run-shell.bash
if ! command -v jq >/dev/null 2>&1; then
    printf '{}\n'
    exit 0
fi
result=$(jq -c --arg runner "$runner" '
    if .hook_event_name != "PreToolUse" or
       ((.tool_name == "Bash" or .tool_name == "exec_command" or
         .tool_name == "functions.exec_command") | not) or
       (.tool_input | type) != "object" then {}
    else
      .tool_input as $input |
      (if ($input.cmd | type) == "string" then "cmd" else "command" end) as $key |
      $input[$key] as $command |
      if ($command | type) != "string" or ($command | length) == 0 or
         ($command | contains("decision_run_shell")) or
         ($command | contains("DECISION_PRUNE_ACTIVE")) then {}
      else
        {hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision: "allow",
          updatedInput: ($input + {($key): (". " + ($runner | @sh) + "; decision_run_shell " + ($command | @sh))})
        }}
      end
    end
' 2>/dev/null) || result='{}'
printf '%s\n' "$result"
