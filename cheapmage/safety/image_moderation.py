"""Offline image moderation.

The moderator works **fully offline**:

* It always runs a dependency-free heuristic analyser (numpy + Pillow only).
* If - and only if - a small local model file already exists on disk under
  ``models/moderation`` (or the directory given via ``model_dir``) it is used
  as an *additional* signal.  Nothing is ever downloaded, and importing this
  module never touches the network or the filesystem beyond a directory
  listing.

Heuristic signals
-----------------
``skin_ratio``
    Fraction of pixels that fall inside a skin-tone gamut (RGB rule set plus
    an HSV band).  High skin coverage is a weak but useful NSFW signal.
``large_skin_area``
    Fraction of the image grid that is dominated by skin pixels.  A single
    large uniform flesh area is a stronger signal than scattered skin pixels.
``edge_density``
    Mean gradient magnitude.  Very smooth, edge-free skin is a hint of
    "over-plastic" / airbrushed rendering, which correlates with synthetic
    NSFW imagery.
``chroma``
    Mean saturation of skin pixels; extremely saturated flesh is unnatural.
``resolution / aspect``
    Degenerate images (tiny, extreme aspect ratio) are flagged as suspect
    output and are never returned.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ModerationResult", "ImageModerator"]


@dataclass(frozen=True)
class ModerationResult:
    """Result of moderating a single image."""

    passed: bool
    nsfw_score: float
    minor_risk: float
    labels: list[str] = field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.passed


#: Scores at or above this value mark the image as failed.
NSFW_THRESHOLD = 0.60
#: Scores at or above this value mark the image as failed (minor risk).
MINOR_RISK_THRESHOLD = 0.75
#: The analyser works on a downscaled copy of this size (fast + enough).
_ANALYSIS_SIZE = 256


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class _HeuristicSignals:
    skin_ratio: float = 0.0
    large_skin_area: float = 0.0
    edge_density: float = 0.0
    skin_saturation: float = 0.0
    skin_value: float = 0.0
    width: int = 0
    height: int = 0


class ImageModerator:
    """Offline NSFW / minor-risk analyser for generated images."""

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str | os.PathLike[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self.enabled = enabled
        self._model: Any = None
        self._model_labels: list[str] = []
        self._model_loaded = False

    # -- local model ------------------------------------------------------
    @staticmethod
    def _default_model_dir() -> Path:
        return Path(__file__).resolve().parent.parent.parent / "models" / "moderation"

    def _load_local_model(self) -> None:
        """Load an optional local ONNX moderation model (never downloads)."""
        if self._model_loaded:
            return
        self._model_loaded = True
        try:
            import onnxruntime  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return
        candidates = [
            self.model_dir / "model.onnx",
            self.model_dir / "moderation.onnx",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            return
        try:  # pragma: no cover - only runs when a model is present
            self._model = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            labels_file = self.model_dir / "labels.txt"
            if labels_file.exists():
                self._model_labels = [
                    line.strip()
                    for line in labels_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        except Exception:  # pragma: no cover - defensive
            self._model = None

    # -- public API -------------------------------------------------------
    def moderate(self, image: Any) -> ModerationResult:
        """Moderate a PIL image and return a :class:`ModerationResult`."""
        if not self.enabled:
            return ModerationResult(
                passed=True,
                nsfw_score=0.0,
                minor_risk=0.0,
                labels=[],
                reason="Image moderation is disabled by configuration.",
            )
        if image is None:
            return ModerationResult(
                passed=False,
                nsfw_score=1.0,
                minor_risk=1.0,
                labels=["missing_image"],
                reason="No image was produced, so it cannot be moderated.",
            )

        signals = self._analyse(image)
        labels: list[str] = []
        nsfw_score = self._score(signals, labels)
        minor_risk = self._minor_risk(signals)

        model_labels, model_score = self._model_signal(image)
        if model_labels:
            labels.extend(model_labels)
        if model_score is not None:
            nsfw_score = max(nsfw_score, model_score)

        degenerate = self._degenerate(labels, signals)
        passed = (
            not degenerate
            and nsfw_score < NSFW_THRESHOLD
            and minor_risk < MINOR_RISK_THRESHOLD
        )

        if passed:
            reason = "Image passed the offline moderation checks."
        elif degenerate:
            reason = (
                "The generated image failed the quality/moderation checks "
                "(degenerate output) and was not returned."
            )
        else:
            reason = (
                "The generated image did not pass the content moderation layer "
                "and was not returned."
            )
        return ModerationResult(
            passed=passed,
            nsfw_score=round(float(nsfw_score), 4),
            minor_risk=round(float(minor_risk), 4),
            labels=labels,
            reason=reason,
        )

    # -- internals --------------------------------------------------------
    @staticmethod
    def _degenerate(labels: list[str], signals: _HeuristicSignals) -> bool:
        return "low_resolution" in labels or "extreme_aspect_ratio" in labels

    @staticmethod
    def _analyse(image: Any) -> _HeuristicSignals:
        """Compute the heuristic signals for ``image``."""
        from PIL import Image  # local import keeps module import cheap

        if not isinstance(image, Image.Image):  # pragma: no cover - defensive
            image = Image.fromarray(np.asarray(image).astype("uint8"))

        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = _ANALYSIS_SIZE / max(width, height)
        if scale < 1.0:
            rgb = rgb.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.BILINEAR,
            )
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return _HeuristicSignals(width=width, height=height)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)

        # --- skin gamut (RGB rule set, Kovac et al. style) ---------------
        rule = (
            (mx > 95)
            & (mn > 40)
            & ((mx - mn) > 15)
            & (r > g)
            & (r > b)
            & ((r - g) > 15)
        )
        # --- skin gamut (HSV band) ---------------------------------------
        delta = mx - mn
        sat = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
        hue = np.zeros_like(mx)
        with np.errstate(invalid="ignore", divide="ignore"):
            h_r = ((g - b) / np.maximum(delta, 1e-6)) % 6
            h_g = ((b - r) / np.maximum(delta, 1e-6)) + 2
            h_b = ((r - g) / np.maximum(delta, 1e-6)) + 4
            hue = np.where(mx == r, h_r, np.where(mx == g, h_g, h_b)) * 60.0
        hue = hue % 360.0
        hsv_rule = (
            (hue <= 45)
            & (sat >= 0.12)
            & (mx >= 45)
            & (r >= g)
        )
        skin = (rule | hsv_rule).astype(np.float32)

        skin_ratio = float(skin.mean())

        # --- large uniform flesh area ------------------------------------
        grid = 16
        h, w = skin.shape
        gh, gw = h // grid, w // grid
        if gh >= 1 and gw >= 1:
            trimmed = skin[: gh * grid, : gw * grid]
            blocks = trimmed.reshape(gh, grid, gw, grid).mean(axis=(1, 3))
            large_area = float((blocks > 0.60).mean())
        else:  # pragma: no cover - tiny images
            large_area = 1.0 if skin_ratio > 0.6 else 0.0

        # --- edge density --------------------------------------------------
        gray = arr.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1)).mean() if w > 1 else 0.0
        gy = np.abs(np.diff(gray, axis=0)).mean() if h > 1 else 0.0
        edge_density = float((gx + gy) / 2.0 / 255.0)

        # --- chroma of the skin pixels -------------------------------------
        if skin_ratio > 0.0:
            skin_sat = float(sat[skin > 0.5].mean()) if skin.any() else 0.0
            skin_val = float(mx[skin > 0.5].mean() / 255.0) if skin.any() else 0.0
        else:
            skin_sat = 0.0
            skin_val = 0.0

        return _HeuristicSignals(
            skin_ratio=skin_ratio,
            large_skin_area=large_area,
            edge_density=edge_density,
            skin_saturation=skin_sat,
            skin_value=skin_val,
            width=width,
            height=height,
        )

    @staticmethod
    def _score(signals: _HeuristicSignals, labels: list[str]) -> float:
        """Combine the heuristic signals into a 0..1 NSFW score."""
        skin_component = _clamp((signals.skin_ratio - 0.12) / 0.38)
        area_component = _clamp((signals.large_skin_area - 0.20) / 0.45)
        chroma_component = _clamp((signals.skin_saturation - 0.45) / 0.35)
        plastic = (
            1.0 if (signals.edge_density < 0.012 and signals.skin_ratio > 0.25) else 0.0
        )

        if skin_component > 0.55:
            labels.append("high_skin_ratio")
        if area_component > 0.55:
            labels.append("large_skin_area")
        if chroma_component > 0.5:
            labels.append("oversaturated_skin")
        if plastic:
            labels.append("over_plastic_skin")
        if signals.width < 64 or signals.height < 64:
            labels.append("low_resolution")
        if signals.width and signals.height:
            ratio = max(signals.width, signals.height) / min(signals.width, signals.height)
            if ratio > 4.0:
                labels.append("extreme_aspect_ratio")

        score = (
            0.42 * skin_component
            + 0.30 * area_component
            + 0.18 * chroma_component
            + 0.10 * plastic
        )
        return _clamp(score)

    @staticmethod
    def _minor_risk(signals: _HeuristicSignals) -> float:
        """Estimate the risk that the depicted subject is a minor.

        A purely visual system cannot answer this reliably; the heuristic is
        intentionally conservative and always combined with the textual
        policy, which is the authoritative source for minor protection.
        """
        skin_component = _clamp((signals.skin_ratio - 0.12) / 0.38)
        area_component = _clamp((signals.large_skin_area - 0.20) / 0.45)
        return _clamp(0.6 * skin_component + 0.4 * area_component) * 0.9

    def _model_signal(self, image: Any) -> tuple[list[str], float | None]:
        """Run the optional local model (if any).  Never downloads anything."""
        self._load_local_model()
        if self._model is None:
            return [], None
        try:  # pragma: no cover - requires a local model file
            from PIL import Image

            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image).astype("uint8"))
            tensor = image.convert("RGB").resize((224, 224), Image.BILINEAR)
            data = np.asarray(tensor, dtype=np.float32) / 255.0
            data = data.transpose(2, 0, 1)[None, ...]
            name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {name: data})
            logits = np.asarray(outputs[0]).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits))
            labels: list[str] = []
            score: float | None = None
            for index, prob in enumerate(probs):
                label = (
                    self._model_labels[index]
                    if index < len(self._model_labels)
                    else f"class_{index}"
                )
                if label.lower() in {"nsfw", "porn", "explicit", "hentai", "nudity"}:
                    score = max(score or 0.0, float(prob))
                    if prob >= 0.5:
                        labels.append(f"model:{label}")
            return labels, score
        except Exception:  # pragma: no cover - defensive
            return [], None
