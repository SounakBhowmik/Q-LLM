from __future__ import annotations

from torch import nn

from quantum_llm.quantum.fake import FakeQuantumLayer
from quantum_llm.quantum.pennylane_layer import PennyLaneQuantumLayer


def build_quantum_layer(backend: str, n_qubits: int, q_layers: int, seed: int) -> nn.Module:
    backend = backend.lower()
    if backend == "fake":
        return FakeQuantumLayer(n_qubits=n_qubits, q_layers=q_layers)
    if backend == "pennylane":
        return PennyLaneQuantumLayer(n_qubits=n_qubits, q_layers=q_layers, seed=seed)
    raise ValueError(f"Unsupported quantum backend '{backend}'. Expected one of: fake, pennylane")
