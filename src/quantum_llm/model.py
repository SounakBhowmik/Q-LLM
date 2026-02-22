from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from quantum_llm.quantum.factory import build_quantum_layer


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    dropout: float
    tie_weights: bool
    quantum: dict
    seed: int


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumAdapter(nn.Module):
    """Projects hidden states to qubit space, applies quantum layer, projects back.

    CPU-friendly mode: pool sequence across T, run one quantum call per batch item,
    then broadcast back over time steps.
    """

    def __init__(self, d_model: int, quantum_cfg: dict, seed: int) -> None:
        super().__init__()
        self.pooling = quantum_cfg.get("pooling", "mean")
        n_qubits = quantum_cfg["n_qubits"]
        self.in_proj = nn.Linear(d_model, n_qubits)
        self.quantum = build_quantum_layer(
            backend=quantum_cfg["backend"],
            n_qubits=n_qubits,
            q_layers=quantum_cfg["q_layers"],
            seed=seed,
        )
        self.out_proj = nn.Linear(n_qubits, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q_in = self.in_proj(x)
        if self.pooling == "mean":
            pooled = q_in.mean(dim=1)
            q_out = self.quantum(pooled)
            q_out = q_out.unsqueeze(1).expand(b, t, -1)
        elif self.pooling == "first_token":
            token = q_in[:, 0, :]
            q_out = self.quantum(token).unsqueeze(1).expand(b, t, -1)
        else:
            raise ValueError(f"Unknown pooling strategy '{self.pooling}'")
        return self.out_proj(q_out)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.block_size)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

        qcfg = cfg.quantum
        self.quantum_adapter: nn.Module | None = None
        if qcfg.get("enabled", False):
            placement = qcfg.get("placement", "every_block")
            every_k = max(1, qcfg.get("every_k_layers", 1))
            use_quantum = placement == "every_block" or (placement == "every_k" and layer_idx % every_k == 0)
            if use_quantum:
                self.quantum_adapter = QuantumAdapter(cfg.d_model, qcfg, seed=cfg.seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if self.quantum_adapter is not None:
            x = x + self.quantum_adapter(x)
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Input sequence length {t} exceeds block_size={self.cfg.block_size}")

        pos = torch.arange(0, t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
