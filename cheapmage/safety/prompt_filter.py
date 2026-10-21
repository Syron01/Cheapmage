"""Pure, dependency-free prompt analysers.

Every public function in this module is a *pure* function: it takes text and
returns a list of matched category names (e.g. ``["minor:child"]``).  Nothing
here reads or writes configuration, performs I/O or touches the network.

Performance
-----------
Blocklists are split into two groups at import time:

* **plain terms** (``"pikachu"``, ``"one piece"``) are matched with a token
  n-gram hash lookup - no regular expressions, no backtracking;
* **pattern terms** (anything containing regex syntax) are matched with a
  single combined alternation regex per category.

A full scan of a 1 kB prompt therefore stays in the low-millisecond range.
"""
from __future__ import annotations

import re
from typing import Sequence

from .blocklists import (
    AMBIGUITY_TERMS,
    CONFIG_OVERRIDE_TERMS,
    COPYRIGHTED_CHARACTER_TERMS,
    EXPLICIT_TERMS,
    MINOR_AGE_PATTERNS,
    MINOR_SEXUAL_TERMS,
    MINOR_TERMS,
    REAL_PERSON_CASE_PATTERNS,
    REAL_PERSON_PATTERNS,
    REAL_PERSON_TERMS,
    SUGGESTIVE_TERMS,
)

__all__ = [
    "detect_minor_terms",
    "detect_explicit_terms",
    "detect_suggestive_terms",
    "detect_real_person",
    "detect_copyrighted_character",
    "detect_config_override_attempt",
    "detect_non_english",
    "detect_age_ambiguity",
    "normalize",
]


# ---------------------------------------------------------------------------
# regex plumbing
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PLAIN_RE = re.compile(r"^[A-Za-z0-9' \-]+$")


def _flex_spaces(term: str) -> str:
    """Turn literal spaces into ``\\s+`` while leaving regex syntax alone.

    ``term`` may itself contain regex constructs (``\\b``, ``(?:a|b)``, ...).
    Only spaces that are *outside* a character class and not escaped are
    treated as separators.
    """
    out: list[str] = []
    in_class = False
    escaped = False
    for ch in term:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "[":
            in_class = True
            out.append(ch)
            continue
        if ch == "]":
            in_class = False
            out.append(ch)
            continue
        if ch == " " and not in_class:
            out.append(r"\s+")
            continue
        out.append(ch)
    return "".join(out)


def _anchor(term: str) -> str:
    """Anchor a term fragment on word boundaries where that makes sense."""
    frag = _flex_spaces(term)
    if frag[:1].isalnum():
        frag = r"\b" + frag
    if frag[-1:].isalnum():
        frag = frag + r"\b"
    return frag


def _plain_key(term: str) -> str:
    """Normalise a plain term into a space-joined token key."""
    return " ".join(_TOKEN_RE.findall(term.lower()))


def _strip_escapes(term: str) -> str:
    """Remove regex escapes so that literal trigger words can be extracted."""
    return term.replace(r"\b", " ").replace(r"\B", " ")


class _Matcher:
    """Fast matcher for one vocabulary.

    Plain terms are looked up in a hash of token n-grams; pattern terms are
    matched with regular expressions.  Pattern terms are additionally indexed
    by the literal words they contain, so a pattern is only executed when at
    least one of its words actually occurs in the text.
    """

    __slots__ = ("_plain", "_max_n", "_groups", "_always")

    def __init__(
        self,
        terms: Sequence[str],
        prefix: str,
        labels: Sequence[str] | None = None,
    ) -> None:
        plain: dict[str, str] = {}
        groups: dict[str, list[tuple[re.Pattern[str], str | None, str]]] = {}
        always: list[tuple[re.Pattern[str], str | None, str]] = []
        max_n = 1
        for index, term in enumerate(terms):
            fixed = labels[index] if labels is not None else None
            if fixed is None and _PLAIN_RE.match(term):
                key = _plain_key(term)
                if key:
                    plain.setdefault(key, f"{prefix}:{term}")
                    max_n = max(max_n, len(key.split()))
                continue
            compiled = re.compile(_anchor(term), re.IGNORECASE)
            words = set(re.findall(r"[a-z]{3,}", _strip_escapes(term)))
            if not words:
                always.append((compiled, fixed, prefix))
                continue
            for word in words:
                groups.setdefault(word, []).append((compiled, fixed, prefix))
        self._plain = plain
        self._max_n = max_n
        self._groups = groups
        self._always = always

    def scan(self, text: str) -> list[str]:
        """Return the labels of every term found in ``text`` (order preserved)."""
        if not text:
            return []
        found: list[str] = []
        seen: set[str] = set()

        tokens = _TOKEN_RE.findall(text.lower())
        if tokens:
            count = len(tokens)
            for size in range(1, self._max_n + 1):
                if size > count:
                    break
                for start in range(0, count - size + 1):
                    key = " ".join(tokens[start : start + size])
                    label = self._plain.get(key)
                    if label is not None and label not in seen:
                        seen.add(label)
                        found.append(label)

            candidates: dict[int, tuple[re.Pattern[str], str | None, str]] = {}
            for token in set(tokens):
                for entry in self._groups.get(token, ()):
                    candidates[id(entry[0])] = entry
            for entry in self._always:
                candidates[id(entry[0])] = entry
            for compiled, fixed, prefix in candidates.values():
                match = compiled.search(text)
                if match is None:
                    continue
                label = fixed if fixed is not None else f"{prefix}:{match.group(0)}"
                if label not in seen:
                    seen.add(label)
                    found.append(label)
        return found


_MINOR = _Matcher(MINOR_TERMS, "minor")
_MINOR_AGE = _Matcher(
    MINOR_AGE_PATTERNS,
    "minor",
    labels=["minor:numeric_age"] * len(MINOR_AGE_PATTERNS),
)
_MINOR_SEXUAL = _Matcher(MINOR_SEXUAL_TERMS, "minor_sexual")
_EXPLICIT = _Matcher(EXPLICIT_TERMS, "explicit")
_SUGGESTIVE = _Matcher(SUGGESTIVE_TERMS, "suggestive")
_REAL_PERSON = _Matcher(REAL_PERSON_TERMS, "real_person")
_COPYRIGHT = _Matcher(COPYRIGHTED_CHARACTER_TERMS, "copyright")
_CONFIG_OVERRIDE = _Matcher(CONFIG_OVERRIDE_TERMS, "config_override")
_AMBIGUITY = _Matcher(AMBIGUITY_TERMS, "ambiguity")

# Real-person wording patterns (case-insensitive and capitalisation sensitive).
_REAL_PERSON_CI_RE = re.compile("|".join(REAL_PERSON_PATTERNS), re.IGNORECASE)
_REAL_PERSON_CASE_RE = re.compile("|".join(REAL_PERSON_CASE_PATTERNS))


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace (used for all keyword scans)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# script detection (English-only policy)
# ---------------------------------------------------------------------------
#: Unicode ranges of scripts the model cannot understand.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0530, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0700, 0x074F, "syriac"),
    (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0E00, 0x0E7F, "thai"),
    (0x3040, 0x309F, "kana"),
    (0x30A0, 0x30FF, "kana"),
    (0x3400, 0x4DBF, "cjk"),
    (0x4E00, 0x9FFF, "cjk"),
    (0xAC00, 0xD7AF, "hangul"),
    (0xF900, 0xFAFF, "cjk"),
)

#: Hard (blocking) non-English categories.
HARD_NON_ENGLISH = frozenset(
    {
        "non_english:greek",
        "non_english:cyrillic",
        "non_english:armenian",
        "non_english:hebrew",
        "non_english:arabic",
        "non_english:syriac",
        "non_english:devanagari",
        "non_english:bengali",
        "non_english:thai",
        "non_english:kana",
        "non_english:cjk",
        "non_english:hangul",
        "non_english:other_script",
    }
)

#: Soft (reported, non-blocking) categories.
SOFT_NON_ENGLISH = frozenset({"non_english:latin_accented", "non_english:symbol"})


def detect_non_english(text: str) -> list[str]:
    """Return categories for non-English content in ``text``.

    Hard categories (non-Latin scripts) mean the prompt must be rejected.
    Soft categories (accented Latin letters, emoji) are reported but do not
    block on their own.
    """
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def add(cat: str) -> None:
        if cat not in seen:
            seen.add(cat)
            found.append(cat)

    for ch in text:
        if not ch.isalpha():
            continue
        code = ord(ch)
        if code < 0x80:
            continue  # plain ASCII letter
        if 0x00C0 <= code <= 0x024F:
            add("non_english:latin_accented")
            continue
        script = "other_script"
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= code <= hi:
                script = name
                break
        add(f"non_english:{script}")
    # Non-ASCII symbols (emoji, CJK punctuation, ...) are reported softly.
    if any(ord(ch) > 0x7F and not ch.isalnum() for ch in text):
        add("non_english:symbol")
    return found


# ---------------------------------------------------------------------------
# public detectors
# ---------------------------------------------------------------------------
def detect_minor_terms(text: str) -> list[str]:
    """Detect references to minors (age words + inherently sexualised terms)."""
    low = normalize(text)
    return _MINOR.scan(low) + _MINOR_AGE.scan(low) + _MINOR_SEXUAL.scan(low)


def detect_explicit_terms(text: str) -> list[str]:
    """Detect explicit (NSFW) sexual vocabulary."""
    return _EXPLICIT.scan(normalize(text))


def detect_suggestive_terms(text: str) -> list[str]:
    """Detect soft/suggestive vocabulary (never blocks on its own)."""
    return _SUGGESTIVE.scan(normalize(text))


def detect_real_person(text: str) -> list[str]:
    """Detect requests for the likeness of a real person."""
    if not text:
        return []
    found = _REAL_PERSON.scan(normalize(text))
    if _REAL_PERSON_CI_RE.search(text):
        for match in _REAL_PERSON_CI_RE.finditer(text):
            found.append(f"real_person_pattern:{match.group(0)}")
    if _REAL_PERSON_CASE_RE.search(text):
        for match in _REAL_PERSON_CASE_RE.finditer(text):
            found.append(f"real_person_pattern:{match.group(0)}")
    return found


def detect_copyrighted_character(text: str) -> list[str]:
    """Detect requests for copyrighted characters or franchises."""
    return _COPYRIGHT.scan(normalize(text))


def detect_config_override_attempt(text: str) -> list[str]:
    """Detect attempts to change safety configuration from prompt text."""
    return _CONFIG_OVERRIDE.scan(normalize(text))


def detect_age_ambiguity(text: str) -> list[str]:
    """Detect wording that leaves the subject's age uncertain."""
    return _AMBIGUITY.scan(normalize(text))
