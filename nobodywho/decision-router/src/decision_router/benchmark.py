"""A fixed set of representative decisions for comparing providers explicitly.

Run with `decision benchmark`. Unlike shadow mode this never runs during real
work, so JEV is only called when the person asks for a benchmark and JEV is
enabled. `expected` is set only where one answer is objectively right; the
ambiguous cases are there to see which providers abstain.
"""

from __future__ import annotations

from typing import Any

CASES: list[dict[str, Any]] = [
    {
        "kind": "failure recovery",
        "expected": "inspect_timezone",
        "state": "A date-parser unit test failed twice with the same one-hour offset after identical "
        "retries. The parser ignores the timezone suffix; fixtures contain '+01:00'.",
        "question": "What should the agent do next?",
        "choices": {
            "retry_same": "Run the same test again unchanged.",
            "inspect_timezone": "Inspect how the parser handles timezone offsets.",
            "rewrite_parser": "Rewrite the parser from scratch.",
        },
    },
    {
        "kind": "failure recovery",
        "expected": "in_repo_venv",
        "state": "`python3 -m pytest` failed twice: 'No module named pytest'. No system-wide or --user "
        "installs allowed. python3 -m venv works; network available.",
        "question": "Which recovery route should the agent take?",
        "choices": {
            "retry_same": "Run the same command again.",
            "in_repo_venv": "Create .venv in the repo and install pytest there.",
            "sudo_install": "Install pytest system-wide with sudo.",
        },
    },
    {
        "kind": "investigation",
        "expected": "edit_and_test",
        "state": "Rename private helper _fmt() to _format_amount(). grep shows 3 call sites, all in "
        "billing/. Billing tests exist and pass.",
        "question": "How much more investigation is worthwhile before editing?",
        "choices": {
            "edit_and_test": "Rename the 3 call sites and run the billing tests.",
            "audit_repo": "Read every module in the repository first.",
            "ask_user": "Stop and ask the user which modules may change.",
        },
    },
    {
        "kind": "debugging",
        "expected": "read_traceback_frame",
        "state": "A request handler returns HTTP 500. The server log has a full traceback ending in "
        "KeyError: 'user_id' at handlers/profile.py:88.",
        "question": "What is the best next debugging step?",
        "choices": {
            "read_traceback_frame": "Open handlers/profile.py around line 88.",
            "add_print_everywhere": "Add print statements across every handler.",
            "restart_server": "Restart the server and hope it goes away.",
        },
    },
    {
        "kind": "delegation",
        "expected": None,
        "state": "Add the same 5-line input guard to 14 independent REST handlers. Subagents are "
        "available; total diff about 70 lines.",
        "question": "How should this work be carried out?",
        "choices": {
            "do_it_inline": "Edit all 14 handlers directly.",
            "subagent_per_handler": "Spawn one subagent per handler.",
            "codemod_script": "Write a one-off codemod, then review its diff.",
        },
    },
    {
        "kind": "implementation route",
        "expected": None,
        "state": "Cache fetch_rate(): entries expire after 60 s, at most 1000 entries, persistence "
        "across restarts is nice-to-have. stdlib only.",
        "question": "Which caching implementation should be used?",
        "choices": {
            "ttl_lru_dict": "Hand-written OrderedDict cache with expiry and LRU eviction.",
            "lru_time_bucket": "functools.lru_cache keyed on a 60-second time bucket.",
            "sqlite_cache": "sqlite3-backed cache with expiry timestamps.",
        },
    },
    {
        "kind": "tool choice",
        "expected": "ripgrep_search",
        "state": "Find every call site of legacy_lookup( in a 4,000-file Python repository.",
        "question": "Which tool should the agent use?",
        "choices": {
            "ripgrep_search": "Run rg -n 'legacy_lookup\\(' over the repo.",
            "open_files_manually": "Open files one by one in an editor.",
            "ask_user": "Ask the user where it is used.",
        },
    },
    {
        "kind": "abstention-worthy",
        "expected": "ABSTAIN",
        "state": "The user asked to 'fix the build'. No build output, error message or failing "
        "command has been provided yet.",
        "question": "Which fix should be applied?",
        "choices": {
            "bump_dependencies": "Upgrade all dependencies.",
            "clear_cache": "Delete the build cache.",
            "pin_compiler": "Pin an older compiler version.",
        },
    },
    {
        "kind": "abstention-worthy",
        "expected": "ABSTAIN",
        "state": "Two libraries could parse the file format. Nothing is known about licence "
        "constraints, performance needs or existing dependencies.",
        "question": "Which library should be adopted?",
        "choices": {"library_a": "Adopt library A.", "library_b": "Adopt library B."},
    },
    {
        "kind": "failure recovery",
        "expected": "break_import_cycle",
        "state": "Tests fail with ImportError: cannot import name 'Item' from partially initialised "
        "module 'app.models' (circular import between app.models and app.services).",
        "question": "Which change should be made?",
        "choices": {
            "retry_tests": "Re-run the tests unchanged.",
            "break_import_cycle": "Move one import inside the function that needs it "
            "(or under TYPE_CHECKING) to break the cycle.",
            "delete_services": "Delete app/services.py.",
        },
    },
]


def requests() -> list[dict[str, Any]]:
    """The cases as DecisionRequest dicts (without the benchmark-only fields)."""
    out = []
    for i, case in enumerate(CASES):
        out.append({
            "id": f"bench-{i:02d}", "question_id": "bench", "state": case["state"],
            "question": case["question"], "choices": case["choices"],
            "allow_abstain": True, "risk": "low",
        })  # fmt: skip
    return out
