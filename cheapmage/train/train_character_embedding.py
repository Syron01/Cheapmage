"""Textual-inversion trainer for CheapMage character embeddings.

    python -m cheapmage.train.train_character_embedding --data data/synth \
        --name ayra --out models/characters --steps 600

What it does
------------
A *character embedding* is a single learned 768-dimensional vector (the
SD-Tiny / CLIP-ViT-L text encoder hidden size) that is bound to one extra
token (e.g. ``<cm_char_ayra>``). At inference the prompt simply contains the
token and the text encoder looks the vector up.

Training follows the classic textual-inversion recipe: the new embedding row
is initialised from the embedding of a neutral class word, and is optimised
so that the encoder's hidden states for a set of captions containing the
placeholder match the hidden states of the same captions with the placeholder
replaced by the class word. Only that one row is trainable - the encoder
itself is frozen.

Output
------
* ``<out>/<name>.safetensors`` - the learned embedding (key
  ``text_model.embeddings.token_embedding.weight``), 768 dims.
* ``<out>/<name>.json`` - metadata: token, placeholder, dims, sha256, licence
  provenance of the training data.

A loader (:func:`load_character_embedding`) is provided so the inference side
can consume the artefact without depending on this module.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import sha256_file
from .train_style_lora import (
    DEFAULT_BASE,
    _seed_everything,
    build_training_dataset,
    resolve_base_snapshot,
)

__all__ = [
    "CharacterEmbedding",
    "load_character_embedding",
    "apply_character_embedding",
    "train_character_embedding",
    "main",
]


@dataclass
class CharacterEmbedding:
    """A learned character token and its embedding vector."""

    token: str
    vector: torch.Tensor  # shape (hidden_size,)
    placeholder: str = ""
    meta: dict[str, Any] | None = None

    @property
    def dims(self) -> int:
        return int(self.vector.shape[-1])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"text_model.embeddings.token_embedding.weight": self.vector.reshape(1, -1)}


def load_character_embedding(path: str | os.PathLike[str]) -> CharacterEmbedding:
    """Load a character embedding saved by this trainer.

    Works with ``<name>.safetensors`` (+ optional ``<name>.json`` metadata)
    and also with the diffusers-style ``*_text_encoder_embeddings.safetensors``
    naming convention.
    """
    from safetensors.torch import load_file

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"character embedding not found: {path}")
    tensors = load_file(str(path))
    key = "text_model.embeddings.token_embedding.weight"
    if key not in tensors:
        candidates = [k for k in tensors if "token_embedding" in k or "embeddings.0" in k]
        if not candidates:
            raise KeyError(f"no embedding tensor found in {path}: {list(tensors)}")
        key = candidates[0]
    vector = tensors[key].reshape(-1)
    meta_path = path.with_suffix(".json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return CharacterEmbedding(
        token=str(meta.get("token", "")),
        vector=vector,
        placeholder=str(meta.get("placeholder", "")),
        meta=meta,
    )


def apply_character_embedding(
    text_encoder: torch.nn.Module,
    tokenizer: Any,
    embedding: CharacterEmbedding,
) -> int:
    """Write ``embedding`` into the text encoder's token embedding table.

    Returns the token id the vector was written to.
    """
    token = embedding.token or embedding.placeholder
    if not token:
        raise ValueError("character embedding has no token")
    ids = tokenizer.convert_tokens_to_ids(token)
    if ids is None or (isinstance(ids, list) and not ids):
        token_id = tokenizer.add_tokens([token])
    else:
        token_id = int(ids[0] if isinstance(ids, list) else ids)
    table = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        table[token_id] = embedding.vector.to(table.dtype).to(table.device)
    return token_id


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_character_embedding(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import CLIPTextModel, CLIPTokenizer

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _seed_everything(args.seed)
    base_path = resolve_base_snapshot(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)

    token = args.token or f"<cm_char_{args.name}>"
    placeholder = args.placeholder or f"a photo of {args.name}"
    class_word = args.class_word

    # Reserve the token and grow the embedding table by one row.
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.to(device)

    table = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        init_ids = tokenizer(
            [class_word],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        init_vec = text_encoder.get_input_embeddings()(init_ids)[0, 0]
        table[token_id] = init_vec
    learnable = torch.nn.Parameter(table[token_id].detach().clone())
    print(f"[ti] token {token!r} -> id {token_id} (dim {table.shape[1]})")

    # A forward hook substitutes the learnable vector wherever the placeholder
    # token appears. The encoder weights stay frozen; only ``learnable`` gets
    # gradients.
    embed_module = text_encoder.get_input_embeddings()

    def _substitute(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        ids = inputs[0]
        mask = (ids == token_id).unsqueeze(-1)
        vector = learnable.to(output.dtype).view(1, 1, -1).expand_as(output)
        return torch.where(mask, vector, output)

    hook_handle = embed_module.register_forward_hook(_substitute)

    # ---- data ------------------------------------------------------------
    dataset, data_summary = build_training_dataset(
        args.data, args.style, args.size, args.max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    optimizer = torch.optim.AdamW([learnable], lr=args.lr, weight_decay=args.weight_decay)
    loss_curve: list[dict[str, float]] = []
    started = time.time()
    step = 0
    epoch = 0

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            captions = list(batch["caption"])
            student_texts = [f"{placeholder}, {caption}" for caption in captions]
            target_texts = [f"{class_word}, {caption}" for caption in captions]

            student_ids = tokenizer(
                student_texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            target_ids = tokenizer(
                target_texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            # The hook injects the learnable vector into the frozen table.
            hidden = text_encoder(student_ids)[0]
            with torch.no_grad():
                target_hidden = text_encoder(target_ids)[0]
            loss = F.mse_loss(hidden.float(), target_hidden.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                print(
                    f"[ti] step {step}/{args.steps} loss {float(loss.detach()):.5f} "
                    f"({(time.time() - started) / step:.2f}s/step)"
                )
                loss_curve.append({"step": step, "loss": float(loss.detach().item())})

    # ---- save ------------------------------------------------------------
    from safetensors.torch import save_file

    vector = learnable.detach().to(torch.float32).cpu().reshape(-1)
    hook_handle.remove()
    tensor_path = out_dir / f"{args.name}.safetensors"
    save_file(
        {"text_model.embeddings.token_embedding.weight": vector.reshape(1, -1).contiguous()},
        str(tensor_path),
    )
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "cheapmage.train.train_character_embedding",
        "name": args.name,
        "token": token,
        "token_id": token_id,
        "placeholder": placeholder,
        "class_word": class_word,
        "dims": int(vector.shape[0]),
        "model": args.base,
        "base_path": str(base_path),
        "steps": step,
        "learning_rate": args.lr,
        "runtime_seconds": round(time.time() - started, 2),
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
        "dataset": data_summary,
        "manifest_sha256": data_summary.get("manifest_sha256", ""),
        "licence_summary": data_summary.get("licence_summary", {}),
        "sha256": sha256_file(tensor_path),
    }
    meta_path = out_dir / f"{args.name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ti] saved {tensor_path} and {meta_path}")
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheapmage.train.train_character_embedding",
        description="Textual inversion: learn a 768-dim character embedding.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--data", required=True)
    parser.add_argument("--style", default="anime")
    parser.add_argument("--name", required=True, help="character name (file name)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--token", default="", help="custom token (default <cm_char_NAME>)")
    parser.add_argument("--placeholder", default="", help="text the token stands for")
    parser.add_argument("--class-word", default="person", help="neutral word to match")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_character_embedding(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
