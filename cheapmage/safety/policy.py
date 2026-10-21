"""Prompt-level safety policy.

The policy is deliberately conservative and *config independent for minors*:
no value of :class:`~cheapmage.config.SafetyConfig` can allow minor-related
sexual content.  Configuration only ever *tightens* the policy, it never
loosens the hard guarantees.

Order of evaluation
-------------------
1. language check (the model only understands English)
2. configuration-override attempt (reported, blocked, config untouched)
3. minor-related sexual content (hard block, no override)
4. real-person likeness (blocked when ``safety.real_person_likeness == "block"``)
5. copyrighted characters (blocked when ``safety.copyrighted_characters == "block"``)
6. explicit content (allowed only when ``nsfw_mode == "on"`` and, if required,
   the caller proved age verification; always blocked when the subject's age
   is ambiguous)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config import SafetyConfig
from .prompt_filter import (
    HARD_NON_ENGLISH,
    detect_age_ambiguity,
    detect_config_override_attempt,
    detect_copyrighted_character,
    detect_explicit_terms,
    detect_minor_terms,
    detect_non_english,
    detect_real_person,
    detect_suggestive_terms,
)

__all__ = ["Decision", "SafetyPolicy"]


@dataclass(frozen=True)
class Decision:
    """Outcome of a safety check.

    Attributes
    ----------
    allowed:
        ``True`` when the request may proceed.
    reason:
        Human readable explanation (safe to show to an end user).
    code:
        Stable machine readable identifier, e.g. ``"minor_sexual"``.
    categories:
        The detector categories that triggered the decision.
    """

    allowed: bool
    reason: str
    code: str
    categories: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.allowed


#: Codes emitted by the policy.
CODE_OK = "ok"
CODE_NON_ENGLISH = "non_english"
CODE_CONFIG_OVERRIDE = "config_override_attempt"
CODE_MINOR = "minor_sexual"
CODE_REAL_PERSON = "real_person"
CODE_COPYRIGHT = "copyrighted_character"
CODE_NSFW_DISABLED = "nsfw_disabled"
CODE_AGE_VERIFICATION = "age_verification_required"
CODE_AGE_AMBIGUITY = "age_ambiguity"
CODE_OUTPUT = "output_moderation_failed"

#: Threshold used when a metadata dict carries a pre-computed NSFW score.
METADATA_NSFW_THRESHOLD = 0.6

#: Labels that mark an output image as unacceptable when present in metadata.
BLOCKING_LABELS = frozenset(
    {
        "nsfw",
        "porn",
        "pornography",
        "explicit",
        "explicit_nudity",
        "nudity",
        "minor",
        "sexual",
        "sexy",
        "suggestive",
        "real_person",
        "celebrity",
        "face",
    }
)


class SafetyPolicy:
    """Applies the CheapMage safety rules to prompts and prompt metadata."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self._config = config if config is not None else SafetyConfig()

    # -- configuration ----------------------------------------------------
    @property
    def config(self) -> SafetyConfig:
        """The (read-only) safety configuration used by this policy."""
        return self._config

    def with_config(self, config: SafetyConfig) -> "SafetyPolicy":
        """Return a new policy bound to ``config`` (never mutates state)."""
        return SafetyPolicy(config)

    # -- prompt checks ----------------------------------------------------
    def check_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        age_verified: bool = False,
    ) -> Decision:
        """Evaluate a prompt (and its negative prompt) against the policy."""
        prompt = prompt or ""
        negative_prompt = negative_prompt or ""
        combined = f"{prompt}\n{negative_prompt}" if negative_prompt else prompt

        # 1. English only -------------------------------------------------
        language = detect_non_english(combined)
        hard_language = [c for c in language if c in HARD_NON_ENGLISH]
        if hard_language:
            return Decision(
                allowed=False,
                reason=(
                    "Unsupported language: the image model only understands English "
                    "prompts. Please rewrite the prompt in English."
                ),
                code=CODE_NON_ENGLISH,
                categories=hard_language,
            )

        # 2. Configuration override attempts ------------------------------
        overrides = detect_config_override_attempt(combined)
        if overrides:
            return Decision(
                allowed=False,
                reason=(
                    "The prompt attempted to change the safety configuration. "
                    "Safety settings are controlled by the administrator only and "
                    "were not modified."
                ),
                code=CODE_CONFIG_OVERRIDE,
                categories=overrides,
            )

        # 3. Minors --------------------------------------------------------
        minor_hits = detect_minor_terms(combined)
        hard_minor = [c for c in minor_hits if c.startswith("minor_sexual:")]
        if hard_minor:
            return Decision(
                allowed=False,
                reason=(
                    "Request blocked: sexualised content involving minors is never "
                    "allowed, regardless of settings."
                ),
                code=CODE_MINOR,
                categories=minor_hits,
            )

        explicit = detect_explicit_terms(prompt)
        suggestive = detect_suggestive_terms(prompt)
        if minor_hits and (explicit or suggestive):
            return Decision(
                allowed=False,
                reason=(
                    "Request blocked: the prompt combines a reference to a minor "
                    "with sexual or suggestive content. This is never allowed."
                ),
                code=CODE_MINOR,
                categories=minor_hits + explicit + suggestive,
            )

        # 4. Real person likeness -----------------------------------------
        if self._config.real_person_likeness == "block":
            people = detect_real_person(combined)
            if people:
                return Decision(
                    allowed=False,
                    reason=(
                        "Request blocked: requests for the likeness of a real person "
                        "are not supported."
                    ),
                    code=CODE_REAL_PERSON,
                    categories=people,
                )

        # 5. Copyrighted characters ---------------------------------------
        if self._config.copyrighted_characters == "block":
            characters = detect_copyrighted_character(combined)
            if characters:
                return Decision(
                    allowed=False,
                    reason=(
                        "Request blocked: requests for copyrighted characters are "
                        "not supported."
                    ),
                    code=CODE_COPYRIGHT,
                    categories=characters,
                )

        # 6. Explicit content ----------------------------------------------
        if explicit:
            if self._config.nsfw_mode != "on":
                return Decision(
                    allowed=False,
                    reason=(
                        "Request blocked: NSFW content is disabled for this "
                        "deployment."
                    ),
                    code=CODE_NSFW_DISABLED,
                    categories=explicit,
                )
            if self._config.nsfw_requires_age_verification and not age_verified:
                return Decision(
                    allowed=False,
                    reason=(
                        "Request blocked: NSFW content is restricted to "
                        f"{self._config.nsfw_allowed_for}. Age verification is "
                        "required."
                    ),
                    code=CODE_AGE_VERIFICATION,
                    categories=explicit,
                )
            ambiguity = detect_age_ambiguity(combined)
            if ambiguity:
                return Decision(
                    allowed=False,
                    reason=(
                        "Request blocked: the subject's age is ambiguous and the "
                        "prompt contains explicit content."
                    ),
                    code=CODE_AGE_AMBIGUITY,
                    categories=explicit + ambiguity,
                )

        return Decision(
            allowed=True,
            reason="Prompt passed the safety policy.",
            code=CODE_OK,
            categories=[],
        )

    # -- output checks ----------------------------------------------------
    def check_output_metadata(self, metadata: Mapping[str, Any] | None) -> Decision:
        """Post-check for prompt-level metadata attached to a produced image.

        The mapping may contain the prompt that was used (``"prompt"`` /
        ``"negative_prompt"``) and/or moderation results produced elsewhere
        (``"nsfw_score"``, ``"labels"``, ``"minor_risk"``).
        """
        if not metadata:
            return Decision(
                allowed=True,
                reason="No output metadata to check.",
                code=CODE_OK,
                categories=[],
            )

        prompt = str(metadata.get("prompt", "") or "")
        negative = str(metadata.get("negative_prompt", "") or "")
        if prompt or negative:
            decision = self.check_prompt(prompt, negative)
            if not decision.allowed:
                return decision

        categories: list[str] = []
        score = metadata.get("nsfw_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if float(score) >= METADATA_NSFW_THRESHOLD:
                categories.append(f"output:nsfw_score={float(score):.2f}")
        minor_risk = metadata.get("minor_risk")
        if isinstance(minor_risk, (int, float)) and not isinstance(minor_risk, bool):
            if float(minor_risk) >= METADATA_NSFW_THRESHOLD:
                categories.append(f"output:minor_risk={float(minor_risk):.2f}")
        labels = metadata.get("labels") or []
        if isinstance(labels, (list, tuple, set)):
            for label in labels:
                if str(label).strip().lower() in BLOCKING_LABELS:
                    categories.append(f"output:label={label}")

        if categories:
            return Decision(
                allowed=False,
                reason=(
                    "Request blocked: the produced image did not pass the content "
                    "moderation layer and was not returned."
                ),
                code=CODE_OUTPUT,
                categories=categories,
            )

        return Decision(
            allowed=True,
            reason="Output metadata passed the safety policy.",
            code=CODE_OK,
            categories=[],
        )
