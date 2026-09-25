"""Render the same concise decision policy into supported global rule locations."""

from __future__ import annotations

import argparse
from pathlib import Path

TARGETS = {
    ".cline/rules/decision-router.md": "cline",
    ".codex/AGENTS.md": "codex",
    ".claude-main/CLAUDE.md": "claude",
    ".claude-max/CLAUDE.md": "csmart",
    ".config/opencode/AGENTS.md": "opencode2",
    ".AGENTS.md": "freebuff",
}
TEMPLATE = Path(__file__).with_name("decision-policy.md")


def install(home: Path) -> list[Path]:
    text = TEMPLATE.read_text()
    written = []
    for relative, caller in TARGETS.items():
        path = home / relative
        rendered = text.replace("@CALLER@", caller)
        if caller == "codex":
            rendered += (
                "\nIn Codex, use the decision-router MCP `decision_ask` tool for bounded "
                "choices; shell calls cannot reach the worker from the command sandbox. "
                "Use `decision_prune_text` for already captured long output.\n"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != rendered:
            backup = path.with_suffix(path.suffix + ".before-decision-policy")
            if not backup.exists():
                backup.write_text(path.read_text())
        path.write_text(rendered)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    args = parser.parse_args()
    for path in install(args.home):
        print(path)


if __name__ == "__main__":
    main()
