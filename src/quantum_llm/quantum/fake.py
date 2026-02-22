from __future__ import annotations

import torch
from torch import nn


class FakeQuantumLayer(nn.Module):
    """Pure-PyTorch stand-in for quantum layers.

    Keeps exactly the same tensor signature as the PennyLane layer,
    so training code can switch backends without changes.
    """

    def __init__(self, n_qubits: int, q_layers: int) -> None:
        super().__init__()
        hidden = max(8, n_qubits * 2)
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_qubits),
        )
        self.scale = nn.Parameter(torch.ones(q_layers, n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, n_qubits]
        out = self.net(x)
        return out * self.scale.mean(dim=0)
