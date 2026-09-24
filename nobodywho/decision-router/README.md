# decision-router (experimental)

An experimental, provider-neutral decision layer for coding agents. The agent asks one narrow, closed
routing question ("which of these next steps?"). The router sends the same sanitized request to one of
two interchangeable backends and returns a normalized, receipted answer:

| Provider    | What it is                                                              | Cost                                     | Confidence reported                      |
| ----------- | ----------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `jev`       | TypeSafe JEV, a hosted decision model (`api.typesafe.ai/v1/systemone`)  | TypeSafe API usage                       | `calibrated_probability` (from TypeSafe) |
| `nobodywho` | A local open-weight GGUF model run by NobodyWho, fully offline          | No inference API fee; uses local CPU/GPU | `sample_stability` (a proxy, see below)  |

This lives in a fork of [NobodyWho](https://github.com/nobodywho-ooo/nobodywho) as an isolated Python
package. It does not change NobodyWho's inference core or bindings. It only uses the public
`nobodywho` Python API (`Model`, `Chat`, `SamplerBuilder`).

**JEV and NobodyWho are different systems.** They are not variants of each other, and their numbers
are not comparable.

## Confidence is not interchangeable

- **JEV** returns `confidence` and a per-option `probabilities` distribution. These are passed through
  unchanged as `confidence_kind: "calibrated_probability"`, with the versioned model that answered
  (for example `jev-1.13.0` for the alias `jev-latest`).
- **NobodyWho** does not expose per-option probabilities through its Python API. The local provider
  never invents a probability or asks the model for a percentage. It decodes each sample under a GBNF
  grammar that admits only the allowed option ids (plus `ABSTAIN` when allowed), so the model cannot
  answer outside the choice set. It draws a small number of seeded samples, permuting option order per
  sample to expose position bias, and reports:
  - `votes`: how many samples chose each id
  - `confidence`: the winner's vote share, labelled `confidence_kind: "sample_stability"`
  - `distribution: null`, because vote shares are not probabilities

  A tie yields no decision (`fallback_reason: "local_no_majority"`).

**A local `sample_stability` of 1.0 is not a TypeSafe calibrated probability of 1.0.** Do not
threshold them against each other.

## Modes

| Mode      | Behaviour                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `off`     | No provider is called.                                                                                                         |
| `jev`     | TypeSafe JEV only.                                                                                                             |
| `local`   | NobodyWho only. The zero-API-cost mode.                                                                                        |
| `shadow`  | JEV is authoritative. NobodyWho gets the identical request, is recorded, and can never set `follow`, even when JEV fails.     |
| `compare` | Both run. `follow` is set only when they agree; on disagreement it is `null` and the note names both answers.                 |

There is deliberately **no automatic local → JEV escalation**. That needs benchmark evidence from
`shadow`/`compare` receipts first.

## Install

```bash
python3 -m venv ~/.local/share/decision-router/venv
~/.local/share/decision-router/venv/bin/pip install nobodywho ./nobodywho/decision-router
ln -s ~/.local/share/decision-router/venv/bin/decision ~/.local/bin/decision

decision local pull      # downloads + pins the default local model (sha256 recorded)
decision install-rule    # installs the Cline rule into ~/.cline/rules/
decision doctor --live   # checks both providers with one real call each
```

The default local model is NobodyWho's own test model, `NobodyWho/Qwen_Qwen3-0.6B-GGUF`
(`Qwen_Qwen3-0.6B-Q4_K_M.gguf`, about 480 MB). It is enough to test the plumbing, but it is too weak
to benchmark decisions against. For shadow benchmarking, pin a stronger model:

```bash
decision local pull --source huggingface:NobodyWho/Qwen_Qwen3-4B-GGUF/Qwen_Qwen3-4B-Q4_K_M.gguf
```

That file is 2.5 GB and needs about 3 GB of RAM, or about 3 GB of VRAM with `"use_gpu": true` in the
`local` section of the config. Routing never downloads. After `pull` the local provider only opens
the pinned file, so it works with no network.

## Local worker

By default (`"persistent": true`) the first local decision starts a worker that keeps the model
loaded, and later decisions reuse it:

- It listens on a Unix socket in `$XDG_RUNTIME_DIR/decision-router/` (mode 0700). There is no
  network port.
- It exits after `idle_timeout_s` (default 900) without jobs.
- It is restarted automatically if it dies.
- A job that exceeds `timeout_s` kills the worker, so the next decision starts clean.

`decision local status` shows the worker and `decision local stop` frees its memory. Set
`"persistent": false` to load the model in a fresh process for every decision.

Measured on a Ryzen 7 3700X / RTX 2070 SUPER with Qwen3-4B Q4_K_M, 3 samples, end to end per
`decision ask`:

| | Fresh process per decision | Persistent worker, warm |
| --- | --- | --- |
| CPU | ~2.9 s | ~1.8 s |
| GPU (Vulkan) | ~1.9 s | ~0.4 s |

## Switching providers

```bash
decision provider            # show the active mode and where it comes from
decision provider off
decision provider jev
decision provider local      # local-only, no inference API fee
decision provider shadow     # JEV decides, local is compared silently
decision provider compare    # explicit A/B testing
DECISION_ROUTER_MODE=local decision ask ...   # one-off override
```

## Asking

```bash
decision ask <<'JSON'
{"question_id": "recovery_route",
 "state": "pytest missing; `python3 -m pytest` failed twice identically; no --user installs",
 "question": "Which recovery route should the agent take?",
 "choices": {"retry_same_command": "Run it again unchanged.",
             "in_repo_venv_install_pytest": "Create .venv in the repo and install pytest there.",
             "rewrite_tests_unittest": "Port the tests to unittest."},
 "allow_abstain": true, "risk": "low"}
JSON
```

The output is one JSON object. The agent acts only on `follow`. It is `null` whenever there is no
usable decision: mode off, bypass, abstention, provider error or timeout, local tie, or a compare-mode
disagreement. The agent then uses its own judgement. A routing failure is always returned as data,
never as a crash.

Contract limits: 2-12 snake_case choices, `ABSTAIN` reserved, `risk` in `low|medium|high`, state under
16k characters. Unknown fields are rejected.

## Safety properties

- **Advisory only.** The router never executes anything. Permissions and execution stay with
  deterministic code and the agent's own checks.
- **Bypass.** `--bypass`, `DECISION_ROUTER_BYPASS=1`, or a user directive containing
  "bypass decision router" / "bypass jev" calls no provider.
- **No recursion.** Calls made while a decision is in flight are refused (`DECISION_ROUTER_ACTIVE`).
  So are questions about whether to use the router itself.
- **Secrets.** The TypeSafe key is read at call time from `~/.config/jev/typesafe.key` (or
  `TYPESAFE_API_KEY`) and only placed in the `Authorization` header. The key and common token
  patterns are redacted from everything a provider sees and from every receipt. The local worker
  subprocess does not inherit the key.
- **Bounded.** JEV calls have a request timeout. Local inference runs in a separate worker process
  with a hard timeout, so a hung or crashed native call cannot take the router down.

## Receipts

Every call appends to `$XDG_STATE_HOME/decision-router/ledger.jsonl` (mode 0600): timestamp, request
id, cwd and repo name, state and payload SHA-256, question id and redacted question, choices, mode,
role (`authoritative` / `shadow` / `compare`), provider, resolved model, choice, abstain, confidence,
`confidence_kind`, distribution or votes, latency, agreement, and error/fallback. Receipts never
contain the state itself, source code, environment or credentials. `decision ledger -n 20` prints a
summary. In shadow and compare mode both receipts share one `request_id` and one `payload_sha256`,
which shows that both providers saw the identical request.

## Development

```bash
cd nobodywho/decision-router
pip install -e '.[dev,local]'
pytest               # offline; the one real-model test is skipped if the GGUF is absent
ruff format --check . && ruff check . && ty check src tests
```

## Keeping the fork current

```bash
git fetch upstream
git switch main && git merge --ff-only upstream/main && git push origin main
git switch decision-router-dev && git rebase main
```

Everything here is confined to `nobodywho/decision-router/`, so upstream merges should not conflict.

## License

NobodyWho is © the NobodyWho contributors and licensed under the [EUPL-1.2](../../LICENSE). This
package is part of a fork of it and is distributed under the same EUPL-1.2 licence.

Built by Cobalt
