#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from quantum_llm.config import deep_update, load_yaml_config, parse_overrides
from quantum_llm.data import build_dataloaders
from quantum_llm.model import ModelConfig, TinyTransformerLM
from quantum_llm.train_utils import make_run_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--override", action="append", default=[], help="dotted.key=value")
    p.add_argument("--dry-run", action="store_true", help="Run 2 optimization steps for smoke test")
    return p.parse_args()


def evaluate(model: TinyTransformerLM, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def main() -> None:
    args = parse_args()
    cfg = deep_update(load_yaml_config(args.config), parse_overrides(args.override))

    seed = int(cfg["seed"])
    set_seed(seed)

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
        seed=seed,
    )

    device = torch.device("cpu")
    model = TinyTransformerLM(mcfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    max_steps = 2 if args.dry_run else cfg["train"]["max_steps"]
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    run_dir = make_run_dir(cfg["train"].get("run_root", "runs"))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    model.train()
    step = 0
    pbar = tqdm(total=max_steps, desc="train")
    while step < max_steps:
        for xb, yb in bundle.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step >= max_steps:
                break

    val_loss = evaluate(model, bundle.val_loader, device)
    print(f"Final val loss: {val_loss:.4f} | ppl: {torch.exp(torch.tensor(val_loss)).item():.3f}")

    ckpt = {
        "model": model.state_dict(),
        "vocab": bundle.tokenizer.stoi,
        "config": cfg,
    }
    torch.save(ckpt, run_dir / "checkpoint.pt")
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
