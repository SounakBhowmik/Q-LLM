#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from quantum_llm.data import build_dataloaders
from quantum_llm.model import ModelConfig, TinyTransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    bundle = build_dataloaders(
        text_path=cfg["data"]["text_path"],
        block_size=cfg["data"]["block_size"],
        batch_size=cfg["train"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["data"].get("num_workers", 0),
    )

    mcfg = ModelConfig(
        vocab_size=bundle.tokenizer.vocab_size,
        block_size=cfg["data"]["block_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        dropout=cfg["model"]["dropout"],
        tie_weights=cfg["model"].get("tie_weights", True),
        quantum=cfg["quantum"],
        seed=cfg["seed"],
    )
    model = TinyTransformerLM(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    losses = []
    with torch.no_grad():
        for xb, yb in bundle.val_loader:
            _, loss = model(xb, yb)
            losses.append(loss.item())

    avg_loss = sum(losses) / max(1, len(losses))
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
