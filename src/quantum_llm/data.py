from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    tokenizer: "CharTokenizer"


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharLMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.tokens = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_text(path: str | Path | None = None) -> str:
    if path is None:
        path = Path("data/tiny_corpus.txt")
    return Path(path).read_text(encoding="utf-8")


def build_dataloaders(
    text_path: str | None,
    block_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int = 0,
) -> DataBundle:
    text = load_text(text_path)
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    split_idx = int(len(token_ids) * (1.0 - val_split))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx - block_size - 1 :]

    train_ds = CharLMDataset(train_ids, block_size=block_size)
    val_ds = CharLMDataset(val_ids, block_size=block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(train_loader=train_loader, val_loader=val_loader, tokenizer=tokenizer)
