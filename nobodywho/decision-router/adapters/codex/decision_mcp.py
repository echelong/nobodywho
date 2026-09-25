"""Small stdio MCP bridge to the shared router for sandboxed Codex sessions.

The bridge exposes data-only tools. It never executes a command supplied by the
agent, so Codex's native command approval and sandbox decisions stay intact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

TOOLS = [
    {
        "name": "decision_ask",
        "description": "Ask the shared local-first router before a meaningful bounded coding choice. Use 2-6 concrete snake_case choice keys; follow the returned follow field when safe.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string"},
                "question": {"type": "string"},
                "choices": {"type": "object", "additionalProperties": {"type": "string"}},
                "allow_abstain": {"type": "boolean"},
            },
            "required": ["state", "question", "choices"],
        },
    },
    {
        "name": "decision_prune_text",
        "description": "Prune already captured long command output with the shared local-first pruner. This does not run commands or alter their approval or exit status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "budget_chars": {"type": "integer", "minimum": 1},
            },
            "required": ["output"],
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "decision_ask":
        payload = {key: arguments[key] for key in ("state", "question", "choices")}
        payload["allow_abstain"] = arguments.get("allow_abstain", True)
        command = ["decision", "ask", "--caller", "codex", "--json", json.dumps(payload)]
        stdin = None
    elif name == "decision_prune_text":
        command = ["decision", "prune", "--caller", "codex", "--json"]
        if "budget_chars" in arguments:
            command.extend(["--budget-chars", str(arguments["budget_chars"])])
        stdin = arguments["output"]
    else:
        raise ValueError(f"unknown tool: {name}")
    result = subprocess.run(
        command, input=stdin, capture_output=True, text=True, timeout=45, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"decision exited {result.returncode}")
    return json.loads(result.stdout)


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    if method and method.startswith("notifications/"):
        return None
    if "id" not in request:
        return None
    request_id = request["id"]
    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "decision-router", "version": "0.1.0"},
                },
            }
        if method == "ping":
            result: dict[str, Any] = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = request.get("params") or {}
            value = call_tool(params["name"], params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(value)}]}
        else:
            raise ValueError(f"unknown method: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": str(exc)},
        }


def main() -> None:
    for line in sys.stdin:
        try:
            response = handle(json.loads(line))
            if response is not None:
                print(json.dumps(response), flush=True)
        except (ValueError, TypeError) as exc:
            print(
                json.dumps(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
