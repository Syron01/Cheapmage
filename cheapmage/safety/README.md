# CheapMage Safety & Moderation

This package implements the content policy of the CheapMage image service.
It has **no network access**, no downloads and no hidden state: everything is
driven by `safety:` in `config/config.yaml`.

```
cheapmage/safety/
├── __init__.py          # public API: SafetyGate, Decision, ModerationResult, ImageModerator
├── policy.py            # Decision + SafetyPolicy (the rules)
├── prompt_filter.py     # pure detector functions (no I/O, no config access)
├── blocklists.py        # the data: term lists and regex fragments
├── image_moderation.py  # offline image analyser (numpy + Pillow)
├── gate.py              # SafetyGate: prompt stage -> generation -> output stage
└── README.md            # this file
```

---

## 1. Hard guarantees

These are **not** configurable and are enforced by `SafetyPolicy`:

| # | Guarantee | Enforcement |
|---|-----------|-------------|
| G1 | Minor-related sexual content is **always** blocked. No prompt, request, config value or API call can enable it. | `policy.check_prompt()` evaluates the minor rules *before* the NSFW rules and ignores `nsfw_mode` entirely. `MINOR_SEXUAL_TERMS` are blocked even without other sexual vocabulary. |
| G2 | Requests for the likeness of a **real person** are blocked (living public figures, `photo of <person>`, `as <celebrity>`, face-swap / deepfake wording). | `policy.check_prompt()` + `blocklists.REAL_PERSON_TERMS` / `REAL_PERSON_PATTERNS`. |
| G3 | Requests for **copyrighted characters** or franchises are blocked (Pikachu, Naruto, Goku, Luffy, Spider-Man, Mickey Mouse, Elsa, Sailor Moon, Gojo, Light Yagami, Tanjiro, Levi, Zelda, Harry Potter, Batman, Gundam, Sonic, ...). | `policy.check_prompt()` + `blocklists.COPYRIGHTED_CHARACTER_TERMS`. |
| G4 | Prompt text can **never** change configuration. Attempts (`enable nsfw`, `nsfw: on`, `turn off the filter`, `ignore safety`, `developer mode`, `ignore previous instructions`, ...) are detected, reported and blocked. | `detect_config_override_attempt()` returns the `config_override_attempt` decision. The policy object holds a *read-only* view of the config and never writes to it. |
| G5 | The model understands **English only**: prompts containing non-Latin scripts are rejected with a clear error. | `detect_non_english()` + `HARD_NON_ENGLISH` in `prompt_filter.py`. |
| G6 | With `moderation_layer: true` **every produced image** is checked; a failing image is **not returned**. | `SafetyGate.check_output()` returns a blocking `Decision` when `ModerationResult.passed` is `False`. |

G1 is verified exhaustively by `tests/test_safety.py::test_no_config_value_can_enable_minor_content`,
which iterates over every combination of `nsfw_mode`, `nsfw_requires_age_verification`,
`real_person_likeness` and `copyrighted_characters`.

---

## 2. `nsfw_mode` semantics

`nsfw_mode` is an **administrator-only** switch (config file or the admin API -
never the prompt).

| `nsfw_mode` | `nsfw_requires_age_verification` | Prompt contains explicit terms | Result |
|-------------|---------------------------------|------------------------------|--------|
| `off` | any | no | allowed |
| `off` | any | yes | **blocked** (`nsfw_disabled`) |
| `on` | `true` | yes, but the caller did not confirm age verification | **blocked** (`age_verification_required`) |
| `on` | `true` | yes, age verified, subject clearly adult | allowed |
| `on` | `false` | yes | allowed |
| `on` | any | yes **and** the subject's age is ambiguous (`barely legal`, `looks young`, ...) | **blocked** (`age_ambiguity`) |

Additional rules that apply regardless of `nsfw_mode`:

* a minor reference **plus** sexual/suggestive vocabulary → blocked (`minor_sexual`);
* inherently sexualised minor terms (`loli`, `shota`, `child porn`, ...) → blocked on their own;
* real-person likeness → blocked when `real_person_likeness: block`;
* copyrighted characters → blocked when `copyrighted_characters: block`.

`blocked_response` (default: *"Request blocked by the content safety policy.
The generated image was not returned."*) is the message returned to the client
for every blocked request.

---

## 3. How to configure

```yaml
safety:
  nsfw_mode: off              # off | on  (admin only!)
  nsfw_requires_age_verification: true
  nsfw_allowed_for: "age-verified adult users only"
  moderation_layer: true      # every produced image is checked
  real_person_likeness: block # block | allow
  copyrighted_characters: block
  blocked_response: "Request blocked by the content safety policy. ..."
```

Runtime changes go through the admin path only:

```python
store.set_nsfw_mode("on")          # authorized admin action
store.update_safety(moderation_layer=True)
```

`ConfigStore.update_safety()` rejects any key outside the allow-list, and the
safety package itself never writes to the configuration.

---

## 4. Using it from the API layer

```python
from cheapmage.safety import SafetyGate, ImageModerator

gate = SafetyGate(lambda: store.config.safety, moderator=ImageModerator())

# --- prompt stage -------------------------------------------------------
decision = gate.check_prompt(prompt, negative_prompt, age_verified=verified)
if not decision.allowed:
    return {"error": gate.blocked_response}     # nothing is generated

# --- generation ---------------------------------------------------------
image = generate(prompt)

# --- output stage -------------------------------------------------------
decision = gate.check_output(image, metadata={"prompt": prompt})
if not decision.allowed:
    return {"error": gate.blocked_response}     # the image is dropped
return {"image": image}
```

`SafetyGate` accepts any config provider that yields a `SafetyConfig`:
a callable, a `ConfigStore`, an object with `.safety` or `.config.safety`, or a
mapping with a `"safety"` key.

`SafetyGate.enforce_prompt()` raises `SafetyBlocked` instead of returning a
decision, for callers that prefer exceptions.

---

## 5. Detection internals

`cheapmage/safety/prompt_filter.py` exposes pure functions that return a list
of matched categories:

| Function | Categories |
|----------|-----------|
| `detect_minor_terms(text)` | `minor:<term>`, `minor:numeric_age`, `minor_sexual:<term>` |
| `detect_explicit_terms(text)` | `explicit:<term>` |
| `detect_suggestive_terms(text)` | `suggestive:<term>` (soft, never blocks alone) |
| `detect_real_person(text)` | `real_person:<name>`, `real_person_pattern:<match>` |
| `detect_copyrighted_character(text)` | `copyright:<term>` |
| `detect_config_override_attempt(text)` | `config_override:<pattern>` |
| `detect_non_english(text)` | `non_english:<script>` |
| `detect_age_ambiguity(text)` | `ambiguity:<pattern>` |

Plain terms are matched with a token n-gram hash lookup; pattern terms are
matched with regular expressions that are pre-indexed by their literal words.
A full prompt check takes well under 2 ms.

The negative prompt is scanned for minors, real persons, copyrighted
characters, override attempts and language, but **not** for explicit terms
(a negative prompt usually *excludes* such content).

---

## 6. Image moderation

`ImageModerator(device="cpu", model_dir=None, enabled=True)` works fully
offline:

* **skin-tone ratio** - RGB + HSV skin gamut coverage;
* **large uniform flesh area** - 16x16 grid analysis (a single big flesh
  region is a much stronger signal than scattered skin pixels);
* **edge density** - very smooth, edge-free skin hints at "over-plastic" /
  airbrushed synthetic imagery;
* **chroma statistics** - unnaturally saturated flesh;
* **resolution / aspect sanity** - degenerate output is never returned.

If (and only if) a local model file already exists under `models/moderation`
(`model.onnx` + `labels.txt`), it is used as an *additional* signal via
`onnxruntime`. Nothing is ever downloaded, and importing the module performs
no I/O beyond a directory listing.

`ModerationResult.passed` is `False` when `nsfw_score >= 0.60` or
`minor_risk >= 0.75`, or when the output is degenerate.

---

## 7. Adding new terms

Edit `cheapmage/safety/blocklists.py` only - the blocklists are plain tuples
of regex-ready strings, so a new term is one line:

```python
COPYRIGHTED_CHARACTER_TERMS = (..., "my new franchise character")
```

Keep the vocabulary free of ordinary English words (no bare `girl`, `boy`,
`young`, `link`, ...), otherwise benign prompts start to be blocked.
