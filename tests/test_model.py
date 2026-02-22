import torch

from quantum_llm.model import ModelConfig, TinyTransformerLM


def _cfg(quantum_enabled: bool = False, backend: str = "fake") -> ModelConfig:
    return ModelConfig(
        vocab_size=20,
        block_size=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        tie_weights=True,
        quantum={
            "enabled": quantum_enabled,
            "backend": backend,
            "n_qubits": 4,
            "q_layers": 2,
            "placement": "every_block",
            "every_k_layers": 1,
            "pooling": "mean",
        },
        seed=1,
    )


def test_forward_shape() -> None:
    model = TinyTransformerLM(_cfg())
    x = torch.randint(0, 20, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 20)
    assert loss is not None


def test_quantum_fake_shape() -> None:
    model = TinyTransformerLM(_cfg(quantum_enabled=True, backend="fake"))
    x = torch.randint(0, 20, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 20)
