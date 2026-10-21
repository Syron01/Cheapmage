"""High level safety gate.

The gate is the single entry point the API layer is expected to use:

.. code-block:: python

    gate = SafetyGate(config_provider=lambda: store.config.safety)
    decision = gate.check_prompt(prompt, negative_prompt, age_verified=verified)
    if not decision.allowed:
        return {"error": gate.blocked_response}

    image = generate(...)
    decision = gate.check_output(image, metadata={"prompt": prompt})
    if not decision.allowed:
        return {"error": gate.blocked_response}   # image is dropped
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from ..config import SafetyConfig
from .image_moderation import ImageModerator, ModerationResult
from .policy import CODE_OUTPUT, Decision, SafetyPolicy

__all__ = ["SafetyGate", "SafetyBlocked"]


class SafetyBlocked(RuntimeError):
    """Raised when a request violates the safety policy."""

    def __init__(self, decision: Decision, message: str = "") -> None:
        self.decision = decision
        self.message = message or decision.reason
        super().__init__(self.message)


class SafetyGate:
    """Orchestrates prompt checks and output moderation."""

    def __init__(
        self,
        config_provider: Callable[[], SafetyConfig] | Any,
        moderator: ImageModerator | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._moderator = moderator

    # -- configuration ----------------------------------------------------
    def _resolve_config(self) -> SafetyConfig:
        provider = self._config_provider
        if callable(provider):
            value = provider()
        else:
            value = provider
        if isinstance(value, SafetyConfig):
            return value
        if isinstance(value, Mapping):
            safety = value.get("safety", value)
            if isinstance(safety, SafetyConfig):
                return safety
            if isinstance(safety, Mapping):
                return SafetyConfig.model_validate(safety)
        safety = getattr(value, "safety", None)
        if isinstance(safety, SafetyConfig):
            return safety
        config = getattr(value, "config", None)
        safety = getattr(config, "safety", None)
        if isinstance(safety, SafetyConfig):
            return safety
        raise TypeError(
            "config_provider must return a SafetyConfig (or an object exposing "
            "'.safety' / '.config.safety')"
        )

    @property
    def safety_config(self) -> SafetyConfig:
        """The current safety configuration (read-only view)."""
        return self._resolve_config()

    @property
    def blocked_response(self) -> str:
        """The message returned to clients when a request is blocked."""
        return self.safety_config.blocked_response

    @property
    def policy(self) -> SafetyPolicy:
        """A policy bound to the current configuration."""
        return SafetyPolicy(self.safety_config)

    @property
    def moderator(self) -> ImageModerator | None:
        return self._moderator

    # -- prompt stage -----------------------------------------------------
    def check_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        age_verified: bool = False,
    ) -> Decision:
        """Run the prompt stage of the gate."""
        return self.policy.check_prompt(
            prompt, negative_prompt=negative_prompt, age_verified=age_verified
        )

    def enforce_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        age_verified: bool = False,
    ) -> Decision:
        """Like :meth:`check_prompt` but raises :class:`SafetyBlocked`."""
        decision = self.check_prompt(prompt, negative_prompt, age_verified)
        if not decision.allowed:
            raise SafetyBlocked(decision, self.blocked_response)
        return decision

    # -- output stage -----------------------------------------------------
    def moderate_image(self, image: Any) -> ModerationResult:
        """Moderate a produced image with the configured moderator."""
        moderator = self._moderator
        if moderator is None:
            moderator = ImageModerator()
            self._moderator = moderator
        return moderator.moderate(image)

    def check_output(self, image: Any = None, metadata: Mapping[str, Any] | None = None) -> Decision:
        """Run the output stage: metadata check + (optional) image moderation.

        When ``safety.moderation_layer`` is enabled every produced image is
        moderated; a failing image is *not* returned (the decision is a
        block).
        """
        config = self.safety_config
        decision = self.policy.check_output_metadata(metadata)
        if not decision.allowed:
            return decision

        if config.moderation_layer and image is not None:
            result = self.moderate_image(image)
            if not result.passed:
                return Decision(
                    allowed=False,
                    reason=result.reason,
                    code=CODE_OUTPUT,
                    categories=list(result.labels),
                )
        return decision

    # -- helpers ----------------------------------------------------------
    def response_for(self, decision: Decision) -> str:
        """The client-facing message for ``decision``."""
        if decision.allowed:
            return ""
        return self.blocked_response

    def run(
        self,
        prompt: str,
        negative_prompt: str = "",
        age_verified: bool = False,
        image: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Decision:
        """Full gate: prompt stage followed by the output stage."""
        decision = self.check_prompt(prompt, negative_prompt, age_verified)
        if not decision.allowed:
            return decision
        return self.check_output(image, metadata)
