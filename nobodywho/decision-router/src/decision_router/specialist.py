"""Tev-style specialist classifier: identity, artifact discovery and verification.

The specialist is a genuinely fine-tuned compact model whose single job is:

    state + question + fixed candidate decisions -> one option id (or ABSTAIN)

under a GBNF option-token grammar with thinking disabled. The training pipeline
lives in `specialist/` at the package root; this module is the runtime side.

Identity is structural and cryptographic, never a rename of a generic model:

- the declared identity (`local-jev-tev-specialist-v1`, family
  `tev-style-specialist`) is bound to a provenance manifest that pins the exact
  artifact digest, the prompt-contract digest and the dataset digests;
- the local worker verifies the SHA-256 of the file it actually loads and the
  provider compares it against the manifest and the config pin;
- any disagreement is `identity_status = "mismatch"` (`PRIMARY_IDENTITY_MISMATCH`
  semantics): it can never be accepted as the specialist and never counts as a
  specialist success.

Unavailable measurements stay unavailable: nobodywho 3.0.0 exposes no token
logits, so `option_logits` / `logit_margin` are always None with a reason.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPECIALIST_ID = "local-jev-tev-specialist-v1"
SPECIALIST_FAMILY = "tev-style-specialist"
SPECIALIST_KIND = "specialized-option-token-classifier"

# Identity statuses. A decision may only be attributed to the specialist when
# the status is VERIFIED: anything else is not the specialist's answer.
VERIFIED = "verified"
MISMATCH = "mismatch"
UNAVAILABLE = "unavailable"
UNVERIFIED = "unverified"

# Fail-closed policy for a wrong artifact. "fail_closed" (default) ends the
# request cleanly; "escalate" is an explicit operator choice to continue at the
# next tier with the mismatch recorded. Neither path may label the result as a
# specialist answer.
IDENTITY_MISMATCH_POLICIES = ("fail_closed", "escalate")

LOGIT_EVIDENCE_UNAVAILABLE = (
    "unavailable: the nobodywho runtime exposes no per-option token logits, so "
    "optionLogits and logitMargin stay null"
)

SHA256_HEX = 64

# The specialist's own classification instruction. The specialization is the
# fine-tuned artifact (see the manifest's training provenance), not this text;
# the prompt is pinned into the manifest so training and runtime can never
# silently diverge.
SPECIALIST_SYSTEM_PROMPT = (
    "You are a decision classifier. You are given one state, one question and a "
    "fixed list of candidate option ids. Output exactly one option id from the "
    "list, or ABSTAIN when it is offered and the state does not justify a "
    "choice. Output only the option id, with no explanation."
)

# Version tag of the shared user-prompt renderer (providers.nobodywho.render_prompt)
# the training pipeline must render with.
PROMPT_RENDERER = "render_prompt.v1"


def system_prompt_sha256() -> str:
    return hashlib.sha256(SPECIALIST_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class ClassifierIdentity:
    """The declared identity of the primary specialist classifier."""

    classifier_id: str = SPECIALIST_ID
    family: str = SPECIALIST_FAMILY
    kind: str = SPECIALIST_KIND
    version: str | None = None
    thinking_enabled: bool = False
    grammar_constrained_option_tokens: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifierId": self.classifier_id,
            "classifierFamily": self.family,
            "classifierKind": self.kind,
            "classifierVersion": self.version,
            "thinkingEnabled": self.thinking_enabled,
            "grammarConstrainedOptionTokens": self.grammar_constrained_option_tokens,
        }


@dataclass
class IdentityCheck:
    """The result of tying the declared identity to the actual artifact."""

    status: str
    reasons: list[str] = field(default_factory=list)
    identity: ClassifierIdentity = field(default_factory=ClassifierIdentity)
    manifest_path: str | None = None
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    verified_by: str | None = None  # "file_digest" | "loaded_artifact_digest" | None

    @property
    def ok(self) -> bool:
        return self.status == VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            **self.identity.to_dict(),
            "manifestPath": self.manifest_path,
            "expectedSha256": self.expected_sha256,
            "observedSha256": self.observed_sha256,
            "verifiedBy": self.verified_by,
        }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """The digest of the artifact actually on disk (streamed; never loads it)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Registration:
    """A specialist registration taken from one tier's `classifier` config block."""

    tier: str
    classifier: ClassifierIdentity
    model_path: str | None
    model_sha256: str | None
    manifest_path: str | None
    on_identity_mismatch: str
    system_prompt_sha256: str | None


def registration_for(tier: str, settings: dict[str, Any]) -> Registration | None:
    """The specialist registration declared by a tier, or None for a generic tier.

    A tier is a specialist tier only when its `classifier` block declares the
    registered specialist id and family. A generic model can never satisfy this
    by being renamed: the manifest-bound artifact digest must back it up, which
    `verify()` enforces.
    """
    block = settings.get("classifier")
    if not isinstance(block, dict):
        return None
    identity = ClassifierIdentity(
        classifier_id=str(block.get("id", "")),
        family=str(block.get("family", "")),
        kind=str(block.get("kind", SPECIALIST_KIND)),
        version=block.get("version"),
        thinking_enabled=bool(block.get("thinking", False)),
        grammar_constrained_option_tokens=bool(block.get("option_token_grammar", True)),
    )
    if identity.classifier_id != SPECIALIST_ID or identity.family != SPECIALIST_FAMILY:
        return None
    if identity.thinking_enabled or not identity.grammar_constrained_option_tokens:
        return None
    policy = str(block.get("on_identity_mismatch", "fail_closed"))
    return Registration(
        tier=str(tier),
        classifier=identity,
        model_path=settings.get("model_path") or None,
        model_sha256=settings.get("model_sha256") or None,
        manifest_path=block.get("manifest") or None,
        on_identity_mismatch=policy if policy in IDENTITY_MISMATCH_POLICIES else "fail_closed",
        system_prompt_sha256=block.get("system_prompt_sha256") or None,
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError("specialist manifest must be a JSON object")
    return data


def prompt_contract_ok(
    manifest: dict[str, Any], system_prompt_sha256: str | None
) -> tuple[bool, str | None]:
    """Whether the runtime prompt contract matches the one trained against."""
    contract = manifest.get("runtime_contract")
    if not isinstance(contract, dict):
        return False, "manifest has no runtime_contract"
    if contract.get("thinking_enabled") is not False:
        return False, "manifest runtime_contract.thinking_enabled must be false"
    if contract.get("option_token_grammar") is not True:
        return False, "manifest runtime_contract.option_token_grammar must be true"
    trained = contract.get("system_prompt_sha256")
    if system_prompt_sha256 and is_sha256(trained) and trained != system_prompt_sha256:
        return False, "the runtime system prompt differs from the trained prompt contract"
    return True, None


def verify(
    registration: Registration,
    *,
    observed_sha256: str | None = None,
    verify_by: str = "declared_pin",
) -> IdentityCheck:
    """Ties the declared specialist identity to the actual artifact.

    `observed_sha256` is the digest of the file the worker actually loaded (when
    the worker measured it) or the on-disk digest (when `verify_by == "file_digest"`).
    Without an observed digest the check can only reach `unverified`.
    """
    reasons: list[str] = []
    check = IdentityCheck(status=UNVERIFIED, identity=registration.classifier)

    if not registration.manifest_path or not os.path.isfile(registration.manifest_path):
        reasons.append("no specialist manifest is registered for this tier")
        check.status, check.reasons = UNAVAILABLE, reasons
        return check
    check.manifest_path = registration.manifest_path
    try:
        manifest = load_manifest(registration.manifest_path)
    except (OSError, ValueError, TypeError) as error:
        reasons.append(f"specialist manifest is unreadable: {error}")
        check.status, check.reasons = UNAVAILABLE, reasons
        return check

    if (
        manifest.get("classifier_id") != SPECIALIST_ID
        or manifest.get("classifier_family") != SPECIALIST_FAMILY
    ):
        reasons.append("manifest classifier identity does not match the registered specialist")
        check.status, check.reasons = MISMATCH, reasons
        return check
    version = manifest.get("version")
    check.identity = ClassifierIdentity(
        classifier_id=SPECIALIST_ID,
        family=SPECIALIST_FAMILY,
        kind=SPECIALIST_KIND,
        version=version if isinstance(version, str) else None,
        thinking_enabled=False,
        grammar_constrained_option_tokens=True,
    )

    artifact = manifest.get("artifact")
    manifest_sha = artifact.get("sha256") if isinstance(artifact, dict) else None
    if not is_sha256(manifest_sha):
        reasons.append("manifest artifact digest is missing or malformed")
        check.status, check.reasons = UNAVAILABLE, reasons
        return check
    check.expected_sha256 = manifest_sha

    if not registration.model_path or not os.path.isfile(registration.model_path):
        reasons.append("the registered specialist artifact file is absent")
        check.status, check.reasons = UNAVAILABLE, reasons
        return check
    if registration.model_sha256 and registration.model_sha256 != manifest_sha:
        reasons.append("the config pin disagrees with the manifest artifact digest")
        check.status, check.reasons = MISMATCH, reasons
        return check

    prompt_ok, prompt_reason = prompt_contract_ok(manifest, registration.system_prompt_sha256)
    if not prompt_ok:
        reasons.append(str(prompt_reason))
        check.status, check.reasons = MISMATCH, reasons
        return check

    if observed_sha256 is not None and not is_sha256(observed_sha256):
        reasons.append("the observed artifact digest is malformed")
        check.status, check.reasons = MISMATCH, reasons
        return check

    if verify_by == "file_digest" and observed_sha256 is None:
        observed_sha256 = sha256_file(registration.model_path)
    check.observed_sha256 = observed_sha256

    if observed_sha256 is None:
        # Structural checks passed but the loaded artifact digest is not known yet.
        check.status, check.reasons = UNVERIFIED, reasons
        return check
    if observed_sha256 != manifest_sha:
        reasons.append("the loaded artifact digest does not match the specialist manifest")
        check.status, check.reasons = MISMATCH, reasons
        return check

    check.status, check.reasons = VERIFIED, reasons
    check.verified_by = "file_digest" if verify_by == "file_digest" else "loaded_artifact_digest"
    return check


def status_block(
    check: IdentityCheck | dict[str, Any] | None, *, tier: Any = None
) -> dict[str, Any]:
    """One honest classifier-status block (shared by outcomes, doctor, discovery).

    `available` is true only for a VERIFIED artifact: a renamed generic model is
    a mismatch, an absent artifact is unavailable, and an unobserved digest is
    unverified — none of them may claim the specialist is available.
    """
    if isinstance(check, dict):
        block = dict(check)
    else:
        if check is None:
            check = IdentityCheck(
                status=UNAVAILABLE, reasons=["no tier registers the Tev specialist classifier"]
            )
        block = check.to_dict()
    block.update(
        available=block.get("status") == VERIFIED,
        tier=tier,
        optionLogits=None,
        logitMargin=None,
        logitEvidence=LOGIT_EVIDENCE_UNAVAILABLE,
    )
    return block


def discovery(
    tiers: dict[str, dict[str, Any]], *, verify_by: str = "file_digest"
) -> dict[str, Any]:
    """The honest primary-classifier status across the configured tiers.

    Finds the first tier that registers the Tev specialist and verifies its
    artifact against the manifest (by default by hashing the file on disk).
    """
    for tier in sorted(tiers):
        registration = registration_for(tier, tiers[tier])
        if registration is None:
            continue
        number = int(tier) if str(tier).isdigit() else tier
        return status_block(verify(registration, verify_by=verify_by), tier=number)
    return status_block(None)
