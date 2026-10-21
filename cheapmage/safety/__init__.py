"""CheapMage safety package.

Public API
----------
:class:`SafetyGate`
    Orchestrates the prompt stage and the output moderation stage.
:class:`Decision`
    Outcome of a prompt/metadata check.
:class:`ModerationResult`
    Outcome of moderating a produced image.
:class:`ImageModerator`
    Offline (network-free) image analyser.
:class:`SafetyPolicy`
    The rules themselves, usable without the gate.

Hard guarantees (see ``README.md`` in this package):

* minor-related sexual content is **always** blocked - no configuration,
  prompt or API call can change that;
* real-person likeness requests are blocked;
* copyrighted character requests are blocked;
* prompt text can never change configuration;
* only English prompts are accepted;
* with ``moderation_layer`` enabled every produced image is checked and a
  failing image is never returned.
"""
from __future__ import annotations

from .gate import SafetyBlocked, SafetyGate
from .image_moderation import ImageModerator, ModerationResult
from .policy import Decision, SafetyPolicy

__all__ = [
    "SafetyGate",
    "SafetyBlocked",
    "Decision",
    "ModerationResult",
    "ImageModerator",
    "SafetyPolicy",
]
