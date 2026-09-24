"""`decision` command line.

decision provider [off|jev|local|shadow|compare]   show or switch the mode
decision ask [--json JSON | --file F] [--mode M] [--bypass] [--user-directive TEXT]
decision doctor [--live]                           check both providers
decision test [--mode M]                           run a canned live decision
decision ledger [-n N]                             recent receipts
decision local pull [--source S]                   download + pin the local model
decision install-rule [--dest DIR]                 install the Cline rule
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from . import config as cfg
from .contract import DecisionRequest, RequestError
from .ledger import Ledger
from .providers import JevProvider, NobodyWhoProvider
from .router import Router

RULE_NAME = "decision-router.md"
DEFAULT_RULE_DIR = Path.home() / ".cline" / "rules"


def build_router(config: dict[str, Any]) -> Router:
    jev = JevProvider.from_config(config)
    local = NobodyWhoProvider.from_config(config)
    literals = jev.secret_literals()
    return Router(
        {"jev": jev, "nobodywho": local},
        Ledger(cfg.ledger_path(), literals),
        literals,
    )


def _print(data: Any) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")


def cmd_provider(args: argparse.Namespace) -> int:
    config = cfg.load()
    if args.mode is None:
        mode, source = cfg.mode(config)
        print(f"{mode}  (from {source})")
        return 0
    cfg.update({"mode": args.mode})
    print(f"decision provider: {args.mode}")
    if args.mode in ("local", "shadow", "compare"):
        ok, why = NobodyWhoProvider.from_config(config).available()
        if not ok:
            print(f"warning: local provider not ready: {why}", file=sys.stderr)
    if os.environ.get("DECISION_ROUTER_MODE"):
        print("warning: DECISION_ROUTER_MODE is set and overrides this", file=sys.stderr)
    return 0


def _read_request(args: argparse.Namespace) -> Any:
    if args.json is not None:
        return json.loads(args.json)
    if args.file is not None:
        return json.loads(Path(args.file).read_text())
    return json.loads(sys.stdin.read())


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        request = DecisionRequest.from_dict(_read_request(args))
    except (ValueError, OSError) as error:
        kind = "invalid_request" if isinstance(error, RequestError) else "unreadable_request"
        _print({"follow": None, "error": kind, "detail": str(error)[:300]})
        return 2
    try:
        config = cfg.load()
        mode = args.mode or cfg.mode(config)[0]
        outcome = build_router(config).route(
            request, mode, bypass=args.bypass, user_directive=args.user_directive
        )
    except Exception as error:  # noqa: BLE001 - the agent gets data, never a traceback
        _print({"follow": None, "error": "router_error", "detail": type(error).__name__})
        return 0
    _print(outcome.to_dict())
    return 0


SAMPLE_REQUEST = {
    "question_id": "next_step",
    "state": (
        "Goal: fix a failing unit test for an ISO-8601 date parser. The test "
        "failed twice with the same one-hour offset after identical retries. "
        "The parser ignores the timezone suffix; the fixtures contain '+01:00'."
    ),
    "question": "What should the agent do next?",
    "choices": {
        "retry_same": "Run the same test again without changing anything.",
        "inspect_timezone": "Inspect how the parser handles timezone offsets before editing.",
        "rewrite_parser": "Rewrite the whole parser from scratch.",
    },
    "allow_abstain": True,
    "risk": "low",
}


def cmd_test(args: argparse.Namespace) -> int:
    config = cfg.load()
    mode = args.mode or cfg.mode(config)[0]
    request = DecisionRequest.from_dict(dict(SAMPLE_REQUEST, id=f"selftest-{os.getpid()}"))
    outcome = build_router(config).route(request, mode)
    print(json.dumps(outcome.to_dict(), indent=2))
    return 0


def _check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'ok' if ok else '!!'}] {label}: {detail}")
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    config = cfg.load()
    mode, source = cfg.mode(config)
    print(f"mode: {mode} (from {source})")
    print(f"config: {cfg.config_path()}")
    print(f"ledger: {cfg.ledger_path()}")
    healthy = _check("mode valid", mode in cfg.MODES, mode)

    print("jev:")
    jev = JevProvider.from_config(config)
    ok, why = jev.available()
    key_file = Path(str(jev.key_file)).expanduser() if jev.key_file else None
    if key_file and key_file.exists():
        perms = stat.S_IMODE(key_file.stat().st_mode)
        _check("key file", perms & 0o077 == 0, f"{key_file} (mode {perms:o}; contents not shown)")
    healthy &= _check("available", ok, why)
    print(f"  endpoint: {jev.endpoint}  model: {jev.model}  timeout: {jev.timeout_s:g}s")

    print("nobodywho:")
    local = NobodyWhoProvider.from_config(config)
    ok, why = local.available()
    local_needed = mode in ("local", "shadow", "compare")
    ready = _check("model", ok, why if not ok else str(local.model_path))
    if local_needed:
        healthy &= ready
    if ok:
        info = local.model_info()
        print(
            f"  model: {info.get('name')}  quantization: {info.get('quantization')}  "
            f"samples: {local.samples}  temperature: {local.temperature}  gpu: {local.use_gpu}"
        )
    try:
        from importlib.metadata import version

        runtime = f"nobodywho {version('nobodywho')}"
        runtime_ok = True
    except Exception:  # noqa: BLE001 - any failure means "not installed"
        runtime, runtime_ok = "nobodywho not installed in this interpreter", False
    runtime_ready = _check("runtime", runtime_ok, runtime)
    if local_needed:
        healthy &= runtime_ready

    print("cline:")
    rule = DEFAULT_RULE_DIR / RULE_NAME
    _check("rule", rule.exists(), str(rule))
    _check(
        "recursion guard",
        os.environ.get("DECISION_ROUTER_ACTIVE") != "1",
        "not inside a decision call",
    )

    if args.live:
        print("live:")
        request = DecisionRequest.from_dict(dict(SAMPLE_REQUEST, id=f"doctor-{os.getpid()}"))
        for name, provider in (("jev", jev), ("nobodywho", local)):
            result = provider.decide(request)
            label = "ABSTAIN" if result.abstain else result.choice
            detail = (
                f"{label} via {result.model} in {result.latency_ms} ms, "
                f"{result.confidence_kind}={result.confidence}"
                if result.ok
                else f"{result.fallback_reason}: {result.error}"
            )
            healthy &= _check(name, result.ok, detail)
    print("healthy" if healthy else "problems found")
    return 0 if healthy else 1


def cmd_ledger(args: argparse.Namespace) -> int:
    for record in Ledger(cfg.ledger_path()).tail(args.n):
        if record.get("skipped"):
            outcome = f"skipped={record['skipped']}"
        else:
            label = "ABSTAIN" if record.get("abstain") else record.get("choice")
            outcome = (
                f"{record.get('provider')}:{record.get('model')} -> {label} "
                f"{record.get('confidence_kind')}={record.get('confidence')} "
                f"{record.get('latency_ms')}ms"
            )
            if record.get("error"):
                outcome += f" error={record.get('fallback_reason')}"
        agreement = record.get("agreement")
        suffix = "" if agreement is None else f" agreement={agreement}"
        print(
            f"{record.get('ts')} {record.get('mode')}/{record.get('role')} "
            f"{record.get('request_id')} {outcome}{suffix}"
        )
    return 0


def cmd_local_pull(args: argparse.Namespace) -> int:
    try:
        import nobodywho
    except ImportError:
        print("nobodywho is not installed in this interpreter", file=sys.stderr)
        return 1
    from .gguf_info import describe

    source = args.source or cfg.load()["local"]["source"]
    path = Path(nobodywho.download_model(source))
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    info = describe(path)
    cfg.update(
        {
            "local": {
                "source": source,
                "model_path": str(path),
                "model_sha256": digest.hexdigest(),
                "model_info": info,
            }
        }
    )
    print(f"local model: {path}")
    print(f"name: {info.get('name')}  quantization: {info.get('quantization')}")
    print(f"sha256: {digest.hexdigest()}")
    return 0


def cmd_install_rule(args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser() if args.dest else DEFAULT_RULE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / RULE_NAME
    with resources.as_file(resources.files("decision_router") / "cline_rule.md") as src:
        shutil.copyfile(src, target)
    print(f"installed {target}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("provider", help="show or switch the decision provider mode")
    s.add_argument("mode", nargs="?", choices=cfg.MODES)
    s.set_defaults(func=cmd_provider)

    s = sub.add_parser("ask", help="route one decision request (JSON on stdin)")
    s.add_argument("--json")
    s.add_argument("--file")
    s.add_argument("--mode", choices=cfg.MODES)
    s.add_argument("--bypass", action="store_true")
    s.add_argument("--user-directive")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("doctor", help="check configuration and providers")
    s.add_argument("--live", action="store_true", help="also call both providers")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("test", help="run a canned live decision through the router")
    s.add_argument("--mode", choices=cfg.MODES)
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("ledger", help="show recent receipts")
    s.add_argument("-n", type=int, default=20)
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("local", help="local model management")
    local = s.add_subparsers(dest="local_command", required=True)
    pull = local.add_parser("pull", help="download and pin the local model")
    pull.add_argument("--source")
    pull.set_defaults(func=cmd_local_pull)

    s = sub.add_parser("install-rule", help="install the Cline rule")
    s.add_argument("--dest")
    s.set_defaults(func=cmd_install_rule)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)
