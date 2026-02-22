# Quantum-Enhanced LLM (Q-LLM)

A minimal, CPU-friendly research scaffold for experimenting with **quantum-inspired/quantum-backed adapters** inside a tiny character-level Transformer language model.

## What "quantum layer" means here

This repo adds a modular `QuantumAdapter` to a standard decoder-only Transformer block:

1. Hidden states `[B, T, D]` are projected to qubit space `[B, T, n_qubits]`.
2. For CPU feasibility, tokens are pooled over sequence length (`mean` by default) to `[B, n_qubits]`.
3. A backend-specific quantum module transforms pooled features.
   - `pennylane`: real PQC with angle embedding, ring entanglement, trainable rotations.
   - `fake`: pure-PyTorch fallback with identical interface.
4. Output is projected back to `[B, T, D]` and residually added.

Because backend selection happens in a factory (`quantum/factory.py`), training code does not change when swapping backends.

## Repository layout

```text
.
├── configs/
│   ├── base.yaml
│   └── quantum.yaml
├── data/
│   └── tiny_corpus.txt
├── scripts/
│   ├── eval.py
│   └── train.py
├── src/quantum_llm/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train_utils.py
│   └── quantum/
│       ├── base.py
│       ├── factory.py
│       ├── fake.py
│       └── pennylane_layer.py
├── tests/
└── pyproject.toml
```

## Quickstart

### 1) Create environment and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Optional (real quantum backend):

```bash
pip install -e .[quantum]
```

### 2) Train vanilla Transformer

```bash
python scripts/train.py --config configs/base.yaml
```

Dry run smoke test:

```bash
python scripts/train.py --config configs/base.yaml --dry-run
```

### 3) Train quantum-enabled model

With PennyLane installed:

```bash
python scripts/train.py --config configs/quantum.yaml
```

Without PennyLane, either install it or override backend to fake:

```bash
python scripts/train.py --config configs/quantum.yaml --override quantum.backend=fake
```

### 4) Evaluate checkpoint

```bash
python scripts/eval.py --checkpoint runs/<timestamp>/checkpoint.pt
```

### 5) Run tests

```bash
pytest -q
```

## Config notes

- `quantum.enabled`: toggles adapter insertion.
- `quantum.placement`: `every_block` or `every_k`.
- `quantum.every_k_layers`: used when `placement=every_k`.
- `quantum.pooling`: `mean` or `first_token` before quantum circuit.
- `quantum.backend`: `pennylane` or `fake`.

CLI override format: `--override model.n_layers=4 --override quantum.enabled=true`

## Swapping quantum backends

Implement a new module with signature `forward(x: Tensor[N, n_qubits]) -> Tensor[N, n_qubits]` and register it in `build_quantum_layer` (`src/quantum_llm/quantum/factory.py`).

Potential targets:
- Qiskit runtime simulator backend
- TorchQuantum layer
- Custom classical surrogate

## Known limitations

- Character-level modeling only (no BPE/tokenizer library integration).
- Quantum call is pooled per sequence for CPU tractability; not per-token full circuit.
- No distributed training, no advanced schedulers, no HF datasets.
- Tiny corpus is for sanity checks, not quality benchmarking.

## Next research steps

1. Add generation script and sampling metrics.
2. Compare placement strategies (`after_attn`, `inside_mlp`, sparse layers).
3. Run ablations on qubit count, circuit depth, pooling method.
4. Introduce richer datasets and tokenizer choices.
5. Add backend benchmarks (speed/quality tradeoff across fake vs PennyLane vs others).
