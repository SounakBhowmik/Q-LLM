import pytest
import torch

from quantum_llm.model import QuantumAdapter


def test_quantum_adapter_fake_backend_shape() -> None:
    adapter = QuantumAdapter(
        d_model=16,
        quantum_cfg={"backend": "fake", "n_qubits": 4, "q_layers": 2, "pooling": "mean"},
        seed=0,
    )
    x = torch.randn(3, 5, 16)
    y = adapter(x)
    assert y.shape == x.shape


def test_quantum_adapter_pennylane_missing_message() -> None:
    with pytest.raises(ImportError, match="PennyLane backend selected"):
        QuantumAdapter(
            d_model=16,
            quantum_cfg={"backend": "pennylane", "n_qubits": 4, "q_layers": 2, "pooling": "mean"},
            seed=0,
        )
