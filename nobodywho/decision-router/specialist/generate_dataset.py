"""Deterministic generator for the Local JEV decision-classification dataset.

The Tev-style specialist learns the Local JEV decision function:

    state + question + fixed candidate decisions -> one option id (or ABSTAIN)

PROVENANCE
  - Examples are SYNTHETIC and generated procedurally from this file's template
    families. No external corpus is copied and no EVOLVE PS.2d/PS.2e outcome is
    ever read or used.
  - LABELS ARE BY CONSTRUCTION: each family defines the correct option as the
    only option satisfying the family's rule over facts stated in the state
    (see FAMILY_RULES). Labels are recorded at generation time from template
    slots. They are never derived from model output, evaluation results or any
    future outcome.
  - SPLIT RULE: an example's split is decided by sha256(family + slot key), so
    every surface rendering of one underlying scenario stays in one split.
  - DEDUPLICATION: exact normalized-text dedup inside each split, and a
    cross-split near-duplicate filter (3-gram shingle Jaccard >= NEAR_DUP_JACCARD
    over state+question text) that drops the later example and counts the drop.
  - The held-out TEST set of approved, human-written Local JEV cases lives in
    `decision_router/benchmark.py` and is NEVER generated here; see
    `specialist/README.md`.

Output: JSONL per split plus `dataset.manifest.json` with dataset version,
digests, example counts, class distributions and the dedup procedure.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "tev-decision-dataset/1.0.0"
DATASET_VERSION = "local-jev-decision-dataset-v1"

SPLITS = ("train", "validation", "test")
SPLIT_SHARES = {"train": 0.75, "validation": 0.125, "test": 0.125}
DEFAULT_COUNTS = {"train": 1150, "validation": 200, "test": 200}

NEAR_DUP_JACCARD = 0.8
ABSTAIN = "ABSTAIN"

FAMILY_RULES = {
    "root_cause_fix": "the state names exactly one explicit cause; the correct option is the only one "
    "that changes that cause",
    "minimal_sufficient_action": "the state says every required fact is already gathered; the correct option is the "
    "smallest action that uses those facts",
    "tool_selection": "the task's stated shape deterministically maps to one tool category; the correct "
    "option is the tool in that category",
    "constraint_match": "the state states one hard constraint; the correct option is the only one that "
    "satisfies it",
    "abstain_insufficient": "the state explicitly withholds every fact that could discriminate between the "
    "options; the only valid output is ABSTAIN",
}

# Neutral option ids: two-word snake_case tokens with no inherent meaning, so a
# model cannot learn "this id is usually right".
ADJECTIVES = (
    "amber", "brisk", "calm", "dim", "eager", "faint", "gentle", "humble", "idle",
    "keen", "lucid", "mild", "nimble", "plain", "quiet", "rapid", "solid", "tidy",
    "vivid", "wary",
)  # fmt: skip
NOUNS = (
    "anchor", "beacon", "cable", "drift", "engine", "fjord", "gadget", "harbor",
    "instrument", "journal", "kettle", "ladder", "meadow", "needle", "outpost",
    "parcel", "quarry", "riddle", "summit", "tunnel",
)  # fmt: skip


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def shingles(text: str, n: int = 3) -> set[str]:
    words = normalized(text).split()
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Slot vocabularies. Every family draws its scenario facts from these pools;
# the split is decided on the slot key, so no surface rendering of one
# underlying scenario can cross splits.
# ---------------------------------------------------------------------------

ROOT_CAUSE = [
    ("a date-parser unit test", "failed twice with the same one-hour offset",
     "the parser ignores the timezone suffix", "handle the timezone suffix in the date parser",
     "the date parser"),
    ("the CSV import job", "crashed with a duplicate primary key",
     "the upsert path inserts instead of updating", "make the upsert path update existing rows",
     "the import job"),
    ("the nightly sync", "timed out after ten minutes",
     "the sync fetches rows one by one instead of in batches",
     "switch the sync to batched row fetches", "the nightly sync"),
    ("the image thumbnailer", "produced zero-byte thumbnails",
     "the resize step closes the output buffer before writing",
     "keep the output buffer open until the thumbnail is written", "the thumbnailer"),
    ("the login integration test", "failed with a stale CSRF token error",
     "the test reuses a token from an earlier session", "mint a fresh CSRF token per test session",
     "the login test"),
    ("the search indexer", "duplicated every document after a redeploy",
     "the indexer never clears its checkpoint before a full rebuild",
     "clear the indexer checkpoint before a full rebuild", "the search indexer"),
    ("the PDF export", "rendered blank pages after the font upgrade",
     "the exporter pins the old font file name", "update the pinned font file name in the exporter",
     "the PDF exporter"),
    ("the webhook retry worker", "delivered every event three times",
     "the retry counter resets on every poll", "persist the retry counter across polls",
     "the retry worker"),
    ("the password reset email", "went to the wrong recipient",
     "the mailer reuses one cached recipient address", "reset the cached recipient per message",
     "the mailer"),
    ("the monthly report job", "double-counted refunded orders",
     "the report sums gross totals without excluding refunds", "exclude refunded orders in the report total",
     "the report job"),
    ("the file upload API", "accepted files larger than the limit",
     "the size check runs after the file is stored", "run the size check before storing the file",
     "the upload API"),
    ("the session sweeper", "logged users out during active use",
     "the sweeper expires sessions by creation time instead of last-seen time",
     "expire sessions by last-seen time in the sweeper", "the session sweeper"),
    ("the currency converter", "rounded every amount down to whole units",
     "the converter truncates instead of rounding half-up", "round half-up in the converter",
     "the converter"),
    ("the audit log writer", "dropped entries under concurrent load",
     "the writer appends to a shared buffer without a lock", "serialize the audit log appends",
     "the audit log writer"),
    ("the DNS cache warmer", "hammered the resolver on every request",
     "the warmer checks freshness but never serves from cache", "serve cached entries within their TTL",
     "the cache warmer"),
    ("the thumbnail queue", "reprocessed the same image endlessly",
     "the queue never records completed jobs", "record completed jobs before dequeuing the next",
     "the thumbnail queue"),
]  # fmt: skip

MINIMAL_ACTION = [
    ("rename the private helper _fmt() to _format_amount()", "grep shows 3 call sites, all in billing/",
     "the billing tests pass", "rename the 3 call sites and run the billing tests",
     "rename the 3 call sites and run the billing tests"),
    ("add an index on orders(created_at)", "the migration is written and reviewed",
     "the staging database is free to migrate now", "run the migration on staging and watch the query plan",
     "run the migration on staging and watch the query plan"),
    ("flip the retry flag to 3 attempts", "the config change is a one-line diff",
     "the integration suite covers the retry path", "apply the one-line diff and run the integration suite",
     "apply the one-line diff and run the integration suite"),
    ("remove the deprecated /v1 alias route", "grep shows the route has no callers",
     "the route tests are the only references left", "delete the route and its tests",
     "delete the route and its tests"),
    ("cache the parsed config object", "the parser is pure and the config is read-only after load",
     "the load path is covered by unit tests", "add the cache and run the config unit tests",
     "add the cache and run the config unit tests"),
    ("fix the off-by-one in the pagination cursor", "the failing case is reproduced in a unit test",
     "the cursor code is 12 lines", "correct the cursor comparison and run the unit tests",
     "correct the cursor comparison and run the unit tests"),
    ("raise the request timeout from 5s to 15s", "the slow endpoint is a single third-party call",
     "the timeout constant is covered by config tests", "raise the constant and run the config tests",
     "raise the constant and run the config tests"),
    ("pin the transitive dependency to 2.4.1", "the breakage appeared exactly when 2.5.0 landed",
     "the lockfile diff is one line", "pin the version in the lockfile and run the suite",
     "pin the version in the lockfile and run the suite"),
    ("delete the unused feature flag", "grep shows the flag is read nowhere",
     "the flag table has a single row left", "delete the flag row and its dead branch",
     "delete the flag row and its dead branch"),
    ("widen the retry backoff cap to 60s", "the incident report shows retries exhausting in 8s",
     "the backoff constants are two named values", "raise the cap and run the retry tests",
     "raise the cap and run the retry tests"),
]  # fmt: skip

TOOL_SELECTION = [
    ("find every occurrence of a literal function name across a 4,000-file repository",
     "a recursive text search tool", "search the repository text for the literal name"),
    ("compute the exact total of one numeric column in a 2 GB CSV file",
     "a streaming column-sum tool", "stream the file and sum that column"),
    ("show what changed between two revisions of one source file",
     "a revision diff tool", "diff the two revisions of the file"),
    ("run the unit tests of one package only",
     "a test runner scoped to one package", "run the tests of just that package"),
    ("find which process is listening on TCP port 8080",
     "a socket inspection tool", "inspect the listening sockets for port 8080"),
    ("measure how long the checkout endpoint takes under load",
     "a load and latency profiling tool", "profile the endpoint under concurrent load"),
    ("extract the second column of every .csv file into one combined file",
     "a batch column-extraction tool", "extract column two from each CSV into one output"),
    ("list the ten largest files in the artifacts directory",
     "a file-size ranking tool", "rank the directory files by size"),
    ("count how often each error code appears in 40 GB of logs",
     "a streaming log aggregation tool", "aggregate the error-code counts over the logs"),
    ("watch a configuration file and re-run tests when it changes",
     "a file-watch trigger tool", "watch the file and re-run tests on change"),
]  # fmt: skip

CONSTRAINT = [
    ("standard library only", "use functools.lru_cache with a time-bucketed key",
     "add the third-party redis client"),
    ("no network access at runtime", "read the lookup table from a bundled file",
     "fetch the lookup table from the vendor API on start-up"),
    ("must survive a crash with no data loss", "write the journal with fsync before acknowledging",
     "buffer the journal in memory and flush on a timer"),
    ("the fix must land as a one-line change", "raise the timeout constant from 5 to 15",
     "refactor the whole retry subsystem"),
    ("read-only for the reporting user", "grant a SELECT-only database role",
     "grant the reporting user the table-owner role"),
    ("must not change the public API shape", "add an optional keyword argument with a default",
     "rename the function and update every caller"),
    ("the job must finish within 10 minutes", "parallelize the per-row work over 8 workers",
     "add a nightly full re-scan of every row"),
    ("no new runtime dependencies", "implement the parser with the existing stdlib stack",
     "pull in a new parsing framework"),
    ("must work offline after install", "bundle the model file inside the package",
     "download the model file on first start-up"),
    ("the migration must be reversible", "ship an explicit down-migration alongside the up",
     "edit the previous migration in place"),
]  # fmt: skip

MISSING_FACT = [
    ("licence constraints", "performance requirements"),
    ("the expected file size range", "the retention requirements"),
    ("the upstream rate limits", "the consistency requirements"),
    ("the team's rollback tooling", "the compliance requirements"),
    ("the peak concurrency", "the durability requirements"),
    ("the browser support matrix", "the accessibility requirements"),
    ("the data residency rules", "the audit requirements"),
    ("the expected request mix", "the availability requirements"),
    ("the budget ceiling", "the staffing requirements"),
    ("the deprecation timeline", "the compatibility requirements"),
]  # fmt: skip

RISKS = ("low", "medium", "high")

# Per-family surface templates and distractor paraphrases: surface diversity
# without touching the label-relevant facts.
ROOT_CAUSE_TEMPLATES = [
    "{subject} {symptom}. The root cause is stated in the report: {cause}.",
    "Report: {subject} {symptom}. Diagnosis: {cause}.",
    "After the latest change, {subject} {symptom}. The log analysis pinned it down: {cause}.",
]
ROOT_CAUSE_DISTRACTORS = [
    [
        "Re-run {subject} without changing anything.",
        "Add more logging around {module} and re-check next week.",
    ],
    [
        "Re-run {subject} exactly as before.",
        "Wait for the next scheduled run and observe {module}.",
    ],
]
MINIMAL_TEMPLATES = [
    "Goal: {goal}. All facts required for the change are already gathered: {facts}; {extra}.",
    "Task: {goal}. The investigation is complete: {facts}; {extra}. No further facts are needed.",
    "{goal} is ready to implement. Known facts (complete): {facts}; {extra}.",
]
MINIMAL_DISTRACTORS = [
    [
        "Read every module in the repository before editing anything.",
        "Stop and ask the user which files may change.",
    ],
    ["Audit the entire code base first.", "Hold the change until someone re-verifies every fact."],
]
TOOL_TEMPLATES = [
    "Task for the coding agent: {task}.",
    "The next job is narrow and specific: {task}.",
    "Work item (single step): {task}.",
]
TOOL_DISTRACTORS = [
    [
        "Open every file in an editor and look through them one by one.",
        "Stop and ask the user which tool to use.",
    ],
    [
        "Manually page through the data with an interactive shell.",
        "Wait for the next planning meeting to pick a tool.",
    ],
]
CONSTRAINT_TEMPLATES = [
    "Hard requirement (not negotiable): {constraint}. Two proposals are on the table.",
    "The change must satisfy this hard constraint: {constraint}. Two proposals exist.",
    "Requirement (hard, already approved): {constraint}.",
]
CONSTRAINT_DISTRACTORS = [
    ["Delay the decision until the constraint is relaxed.", "Flip a coin between the proposals."],
    ["Postpone the choice until requirements change.", "Do nothing this cycle."],
]
MISSING_TEMPLATES = [
    "Two options are both plausible. Nothing is known about {missing1} or {missing2}.",
    (
        "The state contains no information about {missing1} or {missing2}. Both options "
        "remain plausible under what is known."
    ),
    (
        "Nothing in the available evidence speaks to {missing1} or {missing2}; either option "
        "could be right."
    ),
]


def neutral_ids(rng: random.Random, count: int) -> list[str]:
    """Option ids drawn without semantics, so no id can leak the label."""
    ids: list[str] = []
    while len(ids) < count:
        candidate = f"{rng.choice(ADJECTIVES)}_{rng.choice(NOUNS)}"
        if candidate not in ids:
            ids.append(candidate)
    return ids


def _scenario(
    family: str, slot_key: str, state: str | dict, question: str,
    options: dict[str, str], correct: str, rng: random.Random,
) -> dict[str, Any]:  # fmt: skip
    """One candidate scenario: shuffled neutral ids, correct position recorded."""
    count = len(options)
    ids = neutral_ids(rng, count)
    descriptions = list(options.values())
    correct_index = list(options).index(correct) if correct != ABSTAIN else -1
    paired = list(zip(ids, descriptions))
    rng.shuffle(paired)
    shuffled_ids = [i for i, _ in paired]
    shuffled_desc = [d for _, d in paired]
    label = (
        ABSTAIN
        if correct == ABSTAIN
        else shuffled_ids[shuffled_desc.index(descriptions[correct_index])]
    )
    return {
        "family": family,
        "slot_key": slot_key,
        "state": state,
        "question": question,
        "options": dict(zip(shuffled_ids, shuffled_desc)),
        "label": label,
        "label_position": shuffled_ids.index(label) if label != ABSTAIN else len(shuffled_ids),
        "option_count": count,
    }


def build_root_cause(rng: random.Random, slot, combo) -> dict[str, Any]:
    subject, symptom, cause, fix, module = slot
    template, distractor_set, shape, _risk = combo
    text = template.format(subject=subject, symptom=symptom, cause=cause)
    distrs = [
        d.format(subject=subject, module=module) for d in ROOT_CAUSE_DISTRACTORS[distractor_set]
    ]
    state: str | dict = (
        text
        if shape == "prose"
        else {
            "summary": f"{subject} {symptom}",
            "facts": [cause],
        }
    )
    return _scenario(
        "root_cause_fix", f"root_cause:{subject}:{symptom}",
        state, "Which change should be made?",
        {"make_the_change": f"{fix[0].upper()}{fix[1:]}.", "distractor_a": distrs[0],
         "distractor_b": distrs[1]},
        "make_the_change", rng,
    )  # fmt: skip


def build_minimal_action(rng: random.Random, slot, combo) -> dict[str, Any]:
    goal, facts, extra, action, _ = slot
    template, distractor_set, shape, _risk = combo
    text = template.format(goal=goal, facts=facts, extra=extra)
    distrs = [d for d in MINIMAL_DISTRACTORS[distractor_set]]
    state: str | dict = (
        text
        if shape == "prose"
        else {
            "summary": goal,
            "facts": [facts, extra],
            "investigation": "complete",
        }
    )
    return _scenario(
        "minimal_sufficient_action", f"minimal_action:{goal}",
        state, "How much more investigation is worthwhile before acting?",
        {"act_now": f"{action[0].upper()}{action[1:]}.", "distractor_a": distrs[0],
         "distractor_b": distrs[1]},
        "act_now", rng,
    )  # fmt: skip


def build_tool_selection(rng: random.Random, slot, combo) -> dict[str, Any]:
    task, tool_desc, tool_action = slot
    template, distractor_set, shape, _risk = combo
    text = template.format(task=task)
    distrs = [d for d in TOOL_DISTRACTORS[distractor_set]]
    state: str | dict = text if shape == "prose" else {"summary": "single work item", "task": task}
    return _scenario(
        "tool_selection", f"tool:{task}",
        state, "Which tool should the agent use?",
        {"use_tool": f"Use {tool_desc}: {tool_action}.", "distractor_a": distrs[0],
         "distractor_b": distrs[1]},
        "use_tool", rng,
    )  # fmt: skip


def build_constraint_match(rng: random.Random, slot, combo) -> dict[str, Any]:
    constraint, satisfies, violates = slot
    template, distractor_set, shape, _risk = combo
    text = template.format(constraint=constraint)
    distrs = [d for d in CONSTRAINT_DISTRACTORS[distractor_set]]
    state: str | dict = (
        text
        if shape == "prose"
        else {
            "summary": "choose between two proposals",
            "hard_constraint": constraint,
        }
    )
    return _scenario(
        "constraint_match", f"constraint:{constraint}",
        state, "Which proposal should be adopted?",
        {"proposal_a": f"Proposal A: {satisfies}.", "proposal_b": f"Proposal B: {violates}.",
         "distractor_a": distrs[0]},
        "proposal_a", rng,
    )  # fmt: skip


def build_abstain(rng: random.Random, slot, combo) -> dict[str, Any]:
    missing1, missing2 = slot
    template, _distractor_set, shape, _risk = combo
    text = template.format(missing1=missing1, missing2=missing2)
    state: str | dict = (
        text
        if shape == "prose"
        else {
            "summary": "choose between two plausible options",
            "missing": [missing1, missing2],
        }
    )
    return _scenario(
        "abstain_insufficient", f"missing:{missing1}:{missing2}",
        state, "Which option should be chosen?",
        {"option_a": "Adopt the first option as it stands.",
         "option_b": "Adopt the second option as it stands."},
        ABSTAIN, rng,
    )  # fmt: skip


FAMILIES = {
    "root_cause_fix": (ROOT_CAUSE, ROOT_CAUSE_TEMPLATES, 2, build_root_cause),
    "minimal_sufficient_action": (MINIMAL_ACTION, MINIMAL_TEMPLATES, 2, build_minimal_action),
    "tool_selection": (TOOL_SELECTION, TOOL_TEMPLATES, 2, build_tool_selection),
    "constraint_match": (CONSTRAINT, CONSTRAINT_TEMPLATES, 2, build_constraint_match),
    "abstain_insufficient": (MISSING_FACT, MISSING_TEMPLATES, 1, build_abstain),
}
FAMILY_SHARES = {
    "root_cause_fix": 0.30,
    "minimal_sufficient_action": 0.20,
    "tool_selection": 0.20,
    "constraint_match": 0.15,
    "abstain_insufficient": 0.15,
}

SHAPES = ("prose", "object")


def split_for(family: str, slot_index: int) -> str:
    """One slot combination lives in exactly one split, forever.

    Assignment is rank-based inside each family (slots ordered by their hash), so
    every family contributes to every split and the split sizes stay near the
    75/12.5/12.5 target instead of depending on hash luck.
    """
    ranks = _family_ranks(family)
    position = ranks[slot_index]
    n = len(ranks)
    n_val = max(1, round(0.125 * n))
    n_test = max(1, round(0.125 * n))
    if position < n_val:
        return "validation"
    if position < n_val + n_test:
        return "test"
    return "train"


def _family_ranks(family: str) -> dict[int, int]:
    """slot_index -> rank inside its family (deterministic hash order)."""
    if family not in _RANKS:
        slots = range(len(FAMILIES[family][0]))
        ordered = sorted(slots, key=lambda i: sha256_text(f"{family}|{i}"))
        _RANKS[family] = {slot: rank for rank, slot in enumerate(ordered)}
    return _RANKS[family]


_RANKS: dict[str, dict[int, int]] = {}


def combo_key(template: int, distractors: int, shape: str, risk: str) -> str:
    return f"t{template}|d{distractors}|{shape}|{risk}"


def candidates(seed: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Every candidate scenario, split by the frozen slot-key rule."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {s: {} for s in SPLITS}
    for family, (slots, templates, distractor_variants, build) in FAMILIES.items():
        for slot_index, slot in enumerate(slots):
            slot_key = f"{family}:{slot_index}"
            split = split_for(family, slot_index)
            for template in range(len(templates)):
                for distrs in range(distractor_variants):
                    for shape in SHAPES:
                        for risk in RISKS:
                            key = combo_key(template, distrs, shape, risk)
                            rng = random.Random(sha256_text(f"{seed}|{family}|{slot_key}|{key}"))
                            combo = (templates[template], distrs, shape, risk)
                            scenario = build(rng, slot, combo)
                            scenario["risk"] = risk
                            scenario["state_shape"] = shape
                            scenario["combo_key"] = key
                            scenario["slot_index"] = slot_index
                            out[split].setdefault(family, []).append(scenario)
    return out


def select_examples(
    seed: int, counts: dict[str, int]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Deterministic per-split selection with cross-split near-duplicate removal.

    Within a split, surface-similar examples are allowed (they share one slot
    combination by construction). Across splits they are not: a candidate that
    near-duplicates anything already accepted into ANOTHER split is dropped and
    counted.
    """
    pool = candidates(seed)
    chosen: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    shingle_index: dict[str, list[tuple[str, set[str]]]] = {s: [] for s in SPLITS}
    dropped = {"exact_duplicate": 0, "near_duplicate_cross_split": 0, "quota_shortfall": 0}

    def sem_key(scenario: dict[str, Any]) -> str:
        return normalized(json.dumps([scenario["state"], scenario["question"]]))

    def accept(split: str, scenario: dict[str, Any]) -> bool:
        key = sem_key(scenario)
        exact = canonical(
            [scenario["state"], scenario["question"], scenario["options"], scenario["label"]]
        )
        if exact in seen_exact[split]:
            dropped["exact_duplicate"] += 1
            return False
        sh = shingles(key)
        if any(
            jaccard(sh, other) >= NEAR_DUP_JACCARD
            for other_split, entries in shingle_index.items()
            if other_split != split
            for _, other in entries
        ):
            dropped["near_duplicate_cross_split"] += 1
            return False
        seen_exact[split].add(exact)
        shingle_index[split].append((scenario["combo_key"], sh))
        chosen[split].append(scenario)
        return True

    seen_exact: dict[str, set[str]] = {s: set() for s in SPLITS}
    order_rng = random.Random(f"{seed}|selection")
    for split in SPLITS:
        target = counts[split]
        queues = {
            family: sorted(items, key=lambda s: (s["slot_index"], s["combo_key"]))
            for family, items in pool[split].items()
        }
        for items in queues.values():
            order_rng.shuffle(items)
        quotas = {family: max(1, round(FAMILY_SHARES[family] * target)) for family in queues}
        round_robin = sorted(queues) * ((target // max(1, len(quotas))) + 2)
        for family in round_robin:
            if len(chosen[split]) >= target:
                break
            if family not in queues or not queues[family]:
                continue
            if sum(1 for e in chosen[split] if e["family"] == family) >= quotas[family]:
                continue
            accept(split, queues[family].pop())
        for family in sorted(queues):  # top up shortfalls from any family
            while len(chosen[split]) < target and queues[family]:
                accept(split, queues[family].pop())
        if len(chosen[split]) < target:
            dropped["quota_shortfall"] += target - len(chosen[split])
    return chosen, dropped


def render_example(split: str, index: int, scenario: dict[str, Any]) -> dict[str, Any]:
    """One dataset record, rendered with the EXACT runtime prompt renderer."""
    from decision_router.contract import DecisionRequest
    from decision_router.providers.nobodywho import render_prompt

    request_data = {
        "id": f"tev-{split}-{index:05d}",
        "question_id": "decision",
        "state": scenario["state"],
        "question": scenario["question"],
        "choices": scenario["options"],
        "allow_abstain": True,
        "risk": scenario["risk"],
    }
    request = DecisionRequest.from_dict(request_data)
    order = list(scenario["options"]) + [ABSTAIN]
    prompt = render_prompt(request.payload(), order)
    return {
        "id": request_data["id"],
        "split": split,
        "family": scenario["family"],
        "family_rule": FAMILY_RULES[scenario["family"]],
        "slot_key": scenario["slot_key"],
        "combo_key": scenario["combo_key"],
        "risk": scenario["risk"],
        "state_shape": scenario["state_shape"],
        "label": scenario["label"],
        "label_position": scenario["label_position"],
        "label_provenance": "by_construction",
        "request": request_data,
        "train_order": order,
        "prompt": prompt,
        "completion": scenario["label"],
    }


def split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [r["label"] for r in records]
    return {
        "examples": len(records),
        "family_distribution": dict(Counter(r["family"] for r in records)),
        "state_shape_distribution": dict(Counter(r["state_shape"] for r in records)),
        "risk_distribution": dict(Counter(r["risk"] for r in records)),
        "abstain_labels": sum(1 for label in labels if label == ABSTAIN),
        "option_labels": sum(1 for label in labels if label != ABSTAIN),
        "label_position_distribution": dict(Counter(str(r["label_position"]) for r in records)),
    }


def generate(out_dir: Path, seed: int, counts: dict[str, int]) -> dict[str, Any]:
    """Writes the JSONL splits and the dataset manifest; returns the manifest."""
    from decision_router import specialist as spec

    out_dir.mkdir(parents=True, exist_ok=True)
    chosen, dropped = select_examples(seed, counts)
    splits: dict[str, Any] = {}
    for split in SPLITS:
        records = [render_example(split, i, s) for i, s in enumerate(chosen[split])]
        records.sort(key=lambda r: r["id"])
        body = "".join(canonical(r) + "\n" for r in records)
        (out_dir / f"{split}.jsonl").write_text(body)
        splits[split] = {
            **split_stats(records),
            "digest": sha256_text(body),
            "file": f"{split}.jsonl",
        }

    manifest = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256_text(Path(__file__).read_text()),
        "seed": seed,
        "generation_method": "procedural templates over slot vocabularies; surface variants (template "
        "phrasing, fact order, prose vs object state, risk) vary independently of "
        "the label-relevant facts; option ids are neutral two-word tokens sampled "
        "per example and option order is shuffled per example",
        "label_provenance": "by construction: each family's rule over the facts stated in the state "
        "selects exactly one option (or ABSTAIN); labels are written at generation "
        "time from template slots and never come from model output, evaluation "
        "results or EVOLVE outcomes",
        "family_rules": FAMILY_RULES,
        "split_rule": "rank-based per-family slot assignment (slots ordered by sha256(family + "
        "slot index), roughly 75/12.5/12.5 train/validation/test); every surface "
        "rendering of one underlying scenario stays in one split",
        "deduplication": {
            "exact_within_split": "the (state, question, options, label) tuple must be "
            "unique inside a split; examples differing only in "
            "neutral option ids or option order are retained on "
            "purpose (id/order-invariance augmentation)",
            "near_duplicate_cross_split": f"3-gram shingle Jaccard >= {NEAR_DUP_JACCARD} over normalized "
            "state+question text against an accepted example of any other split "
            "drops the later example (processing order: train, validation, test)",
            "scenario_split_rule_enforced_on_top": "slot-key split assignment above",
            "dropped": dropped,
        },
        "runtime_contract": {
            "system_prompt": spec.SPECIALIST_SYSTEM_PROMPT,
            "system_prompt_sha256": spec.system_prompt_sha256(),
            "renderer": spec.PROMPT_RENDERER,
            "thinking_enabled": False,
            "option_token_grammar": True,
        },
        "splits": splits,
        "notes": [
            (
                "The approved human-written held-out TEST cases live in "
                "src/decision_router/benchmark.py and are never generated here; the final "
                "benchmark compares the specialist against them untouched."
            ),
            (
                "This dataset trains the Local JEV decision function (state + question + "
                "fixed options -> one option id). It contains no trading intelligence, no "
                "profitability labels and no EVOLVE PS.2d/PS.2e outcomes."
            ),
        ],
    }
    (out_dir / "dataset.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "dataset")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--validation", type=int, default=DEFAULT_COUNTS["validation"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    args = parser.parse_args()
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    manifest = generate(args.out_dir, args.seed, counts)
    print(
        json.dumps(
            {"splits": manifest["splits"], "dropped": manifest["deduplication"]["dropped"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
