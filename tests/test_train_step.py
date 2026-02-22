import torch

from quantum_llm.model import ModelConfig, TinyTransformerLM


def test_single_training_step_runs() -> None:
    cfg = ModelConfig(
        vocab_size=30,
        block_size=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        tie_weights=False,
        quantum={"enabled": False, "backend": "fake", "n_qubits": 4, "q_layers": 2},
        seed=42,
    )
    model = TinyTransformerLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 30, (4, 8))
    logits, loss = model(x, x)
    assert logits.shape == (4, 8, 30)
    assert loss is not None
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
