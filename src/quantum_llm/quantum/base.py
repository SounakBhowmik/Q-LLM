from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantumConfig:
    enabled: bool = False
    backend: str = "fake"
    n_qubits: int = 4
    q_layers: int = 2
    placement: str = "every_block"
    every_k_layers: int = 1
    pooling: str = "mean"
