#!/usr/bin/bash
# Source this from the shell that Codex already started, then call
# decision_run_shell with the original command as one argument.

decision_run_shell() (
    local command_text="$1" original pruned status chars threshold config_file
    if [ "${DECISION_PRUNE_ACTIVE:-}" = 1 ]; then
        eval "$command_text"
        exit $?
    fi
    umask 077
    original=$(mktemp "${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/decision-run.XXXXXXXX") || {
        eval "$command_text"
        exit $?
    }
    pruned=$(mktemp "${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/decision-pruned.XXXXXXXX") || {
        rm -f -- "$original"
        eval "$command_text"
        exit $?
    }
    trap 'rm -f -- "$original" "$pruned"' EXIT
    ( export DECISION_PRUNE_ACTIVE=1; eval "$command_text" ) > "$original"
    status=$?
    chars=$(wc -m < "$original")
    config_file="${DECISION_ROUTER_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/decision-router}/config.json"
    threshold=$(jq -r '.prune.min_output_chars // 16000' "$config_file" 2>/dev/null) || threshold=16000
    case "$threshold" in
        ''|*[!0-9]*) threshold=16000 ;;
    esac
    if [ "$chars" -le "$threshold" ]; then
        cat -- "$original"
    elif command -v decision >/dev/null 2>&1 && decision prune --caller codex < "$original" > "$pruned"; then
        cat -- "$pruned"
    else
        cat -- "$original"
    fi
    exit "$status"
)
