from __future__ import annotations

import torch
from torch import nn

try:
    import pennylane as qml
except ImportError as exc:  # handled during construction
    qml = None
    _PL_IMPORT_ERROR = exc
else:
    _PL_IMPORT_ERROR = None


class PennyLaneQuantumLayer(nn.Module):
    """Parameterized quantum circuit with angle embedding + entangling ring.

    Input shape: [N, n_qubits]
    Output shape: [N, n_qubits] (Pauli-Z expectations)
    """

    def __init__(self, n_qubits: int, q_layers: int, seed: int = 42) -> None:
        super().__init__()
        if qml is None:
            raise ImportError(
                "PennyLane backend selected but pennylane is not installed. "
                "Install extras: pip install '.[quantum]'"
            ) from _PL_IMPORT_ERROR

        self.n_qubits = n_qubits
        self.q_layers = q_layers

        torch.manual_seed(seed)
        self.weights = nn.Parameter(0.01 * torch.randn(q_layers, n_qubits, 3))

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            for layer in range(q_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self._circuit(sample, self.weights) for sample in x]
        return torch.stack(outputs, dim=0)
