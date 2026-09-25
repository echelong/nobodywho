#!/usr/bin/bash
# Codex only accepts updatedInput together with permissionDecision: allow.
# Until it can preserve the original approval decision, leave tool calls intact.
printf '{}\n'
