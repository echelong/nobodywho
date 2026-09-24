# decision-router (experimental)

An experimental, provider-neutral support layer for coding agents. The coding client still performs
the coding task. For a narrow, closed routing question ("which of these next steps?"), the shared
router sends the same sanitized request through local tiers first and returns a normalized, receipted
answer:

| Provider    | What it is                                                              | Cost                                     | Confidence reported                      |
| ----------- | ----------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `jev`       | TypeSafe JEV, a hosted decision model (`api.typesafe.ai/v1/systemone`)  | TypeSafe API usage                       | `calibrated_probability` (from TypeSafe) |
| `nobodywho` | A local open-weight GGUF model run by NobodyWho, fully offline          | No inference API fee; uses local CPU/GPU | `sample_stability` (a proxy, see below)  |

Production mode is `local-first`: NobodyWho Tier 1, NobodyWho Tier 2 when Tier 1 is unacceptable,
then JEV only when both local tiers fail and the global JEV switch is enabled. The default config
keeps that switch off.

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
| `local-first` | NobodyWho Tier 1, then Tier 2 for an unacceptable result, then JEV only when both local tiers fail and JEV is enabled. |

The local-first acceptance policy uses `sample_stability`, never a calibrated probability. It
escalates for provider errors, timeouts, invalid output, abstention, no valid choice, insufficient
evidence, or stability below the configured threshold. Disagreement or a surprising answer alone
does not trigger another tier.

## Install

```bash
python3 -m venv ~/.local/share/decision-router/venv
~/.local/share/decision-router/venv/bin/pip install nobodywho ./nobodywho/decision-router
ln -s ~/.local/share/decision-router/venv/bin/decision ~/.local/bin/decision

decision local pull      # downloads + pins the default local model (sha256 recorded)
decision install-rule    # installs the Cline rule into ~/.cline/rules/
decision doctor          # checks configuration and local model availability without live calls
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

Each worker also stops early:

- a sample stops as soon as the text generated so far can only become one option id (the grammar
  forces the rest), which usually means one decoded token instead of three to five;
- sampling stops once the remaining samples could not change the winner or whether it passes the
  acceptance policy (with the default 3 samples and `min_stability` 0.66, two agreeing samples
  settle it). The reported `sample_stability` is then the winner's votes over *all planned*
  samples, so an early stop never reports more stability than a full run could. Set
  `"early_stop": false` on a tier to always draw every sample.

Before a GPU load the worker checks free VRAM (NVIDIA): NobodyWho offloads as many layers as fill
free VRAM without reserving room for the context, so a model that does not fully fit fails with an
out-of-memory error rather than running partially offloaded. A model that does not fit is loaded on
CPU instead, or, with `"cpu_fallback": false`, the tier fails fast so the next tier runs. Each GPU
tier stops the other tier's GPU worker before it cold-starts (`evict_tiers`), because an 8 GB card
holds the 4B or the 9B, not both.

## Latency

`decision benchmark latency` measures every call the way a coding client makes it: a fresh
`decision` process (start-up, config, socket, persistent worker, inference, validation, ledger)
against a private temporary config in which the measured model is tier 1. It never contacts
TypeSafe and does not touch the real config, ledger or workers.

```bash
decision benchmark latency                       # the configured models
decision benchmark latency --installed           # every cached GGUF
decision benchmark latency --model /path/to.gguf --operation prune --sizes small,medium
```

Measured 2026-09-24 on a Ryzen 7 3700X / 32 GB / RTX 2070 SUPER (8 GB, Vulkan), NobodyWho 3.0.0,
warm persistent worker, end to end per call (the desktop and other applications were running):

| Model (Q4_K_M) | Operation | Cold | p50 | p95 | p99 | GPU offload | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3 4B | decision | 1.6 s | 278 ms | 308 ms | 339 ms | 37/37 | 16/16 |
| Qwen3 4B | prune small (17-20k chars) | 1.9 s | 302 ms | 497 ms | 590 ms | 37/37 | facts 9/9 |
| Qwen3 4B | prune medium (80-85k) | | 349 ms | 600 ms | 610 ms | 37/37 | facts 10/10 |
| Qwen3 4B | prune large (0.6-0.7M) | | 870 ms | | | 37/37 | facts 2/2 |
| Qwen3 4B | prune judgement | | 208 ms | | | | useful kept 9/9, noise dropped 2/10 |
| Qwen3.5 9B | decision | 3.2 s | 537 ms | 772 ms | 783 ms | 33/33 | 16/16 |
| Qwen3.5 9B | prune small / medium / large | 3.5 s | 1153 / 1161 / 1473 ms | | | 33/33 | facts all kept |
| Qwen3.5 9B | prune judgement | | 498 ms | | | | useful kept 9/9, noise dropped 10/10 |
| Qwen3 0.6B | decision / prune judgement | 1.0 s | 180 / 93 ms | | | 29/29 | 10/16; useful kept 4/9 (fails) |
| Qwen3.5 0.8B | decision / prune judgement | 4.9 s | 238 / 246 ms | | | 25/25 | 6/16; useful kept 3/9 (fails) |
| Qwen3.6 27B | any | | | | | does not fit | GPU load fails (out of VRAM); CPU only |

Before this tuning the same 4B took 494 ms p50 per decision and 1.1-1.5 s / 4.3-4.9 s /
5.2-5.8 s to prune small / medium / large outputs; the 9B took 1044 ms per decision.

Chosen tiers, per operation:

- **decision**: tier 1 Qwen3 4B (GPU, resident), tier 2 Qwen3.5 9B (GPU, cold-starts after
  evicting tier 1, `cpu_fallback: false`), tier 3 JEV only when enabled. The 9B was as accurate
  as the 4B on the benchmark but twice as slow.
- **prune**: tier 1 Qwen3 4B, tier 2 Qwen3.5 9B, JEV only when enabled, then native truncation.
  The 9B judges noise better, but at 1.1-1.5 s it misses the 500 ms target; the 4B never dropped
  a useful block. The 0.6B and 0.8B models are faster but drop useful blocks.
- The 27B is not in the interactive path: on this card it cannot be partially offloaded with
  NobodyWho 3.0.0, and on CPU a call takes tens of seconds.

## Switching providers

```bash
decision provider            # show the active mode and where it comes from
decision provider off
decision provider jev
decision provider local      # local-only, no inference API fee
decision provider shadow     # JEV decides, local is compared silently
decision provider compare    # explicit A/B testing
decision provider local-first
decision jev status          # global TypeSafe kill switch state
decision jev disable         # hard block on TypeSafe calls in every mode
decision benchmark           # explicit local decision and pruning quality benchmark
decision benchmark latency   # end-to-end latency per local model and operation (never JEV)
DECISION_ROUTER_MODE=local decision ask ...   # one-off override
```

`decision provider` prints both the decision and pruning tiers. Use `decision jev enable` only when
TypeSafe fallback is intentionally allowed. `decision jev disable` blocks it centrally even if a
client still has an old disabled plugin on disk.

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

## Pruning command output

`decision prune` is a separate operation from `decision ask`. It keeps command output extractive,
preserves error text, file and line references, test failures, commands, warnings, diff headers and
counts, and marks omitted spans. It tries Tier 1, Tier 2 after a quality or provider failure, JEV
only after both local tiers fail and the global switch is on, then native bounded truncation. A prune
failure never changes the command's exit status or stderr.

Most of the work is deterministic: colour codes are removed, runs of identical lines keep one copy,
every line is matched against the critical-line rules once, and repetitive progress is dropped
without a model. A local tier is then asked about only the blocks that could still fit in the
budget, nearest the start and end of the output first (at most `max_blocks`, default 8), in **one
prompt and one generation** that answers `keep` or `drop` per block. No samples are repeated and no
text is generated. When the critical lines already fill the budget, no model is called.

```bash
decision prune --caller codex -- bash -c 'pytest -q'
```

The one user config is `~/.config/decision-router/config.json`; all client adapters use the same
router and append metadata-only receipts to `~/.local/state/decision-router/ledger.jsonl`. Original
command output archives are off by default because logs can contain credentials.

## Safety properties

- **Advisory only.** The router never executes anything. Permissions and execution stay with
  deterministic code and the agent's own checks.
- **Bypass.** `--bypass`, `DECISION_ROUTER_BYPASS=1`, or a user directive containing
  "bypass decision router" / "bypass jev" calls no provider.
- **No recursion.** Calls made while a decision is in flight are refused (`DECISION_ROUTER_ACTIVE`).
  So are questions about whether to use the router itself.
- **Secrets.** Only the shared router reads the TypeSafe key from `~/.config/jev/typesafe.key` (or
  `TYPESAFE_API_KEY`) when JEV is enabled and actually invoked. Coding client launchers remove that
  environment variable. The key and common token
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
