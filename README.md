# Distributed Transformer

A from-scratch implementation of LLaMA 3 with Ring Attention for context parallelism, enabling training and inference on very long sequences by distributing attention computation across multiple GPUs.

## Features

- **LLaMA 3 Architecture**: Complete implementation with RMSNorm, SwiGLU, RoPE, and Grouped Query Attention (GQA)
- **Ring Attention**: Context parallelism via ring topology for K,V rotation with online softmax
- **Memory Efficient**: Each GPU only holds 1/N of the sequence, enabling N× longer contexts
- **Communication Hiding**: Async send/recv overlaps with compute for near-vanilla latency

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ring Attention                          │
│                                                             │
│   GPU 0        GPU 1        GPU 2        GPU 3              │
│  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐            │
│  │Q[0] │      │Q[1] │      │Q[2] │      │Q[3] │  Q stays   │
│  │K[0]│──────►│K[0]│──────►│K[0]│──────►│K[0]│  K,V rotate │
│  │V[0]│       │V[0]│       │V[0]│       │V[0]│             │
│  └─────┘      └─────┘      └─────┘      └─────┘            │
│     │            │            │            │                │
│     └────────────┴────────────┴────────────┘                │
│              Ring communication (NVLink/PCIe)               │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
distributed_transformer/
├── model/
│   ├── llama.py           # LLaMA model: Transformer, CausalSelfAttention, MLP, RMSNorm
│   └── ring_attention.py  # Ring attention forward pass with online softmax
├── tests/
│   ├── test_ring_attention.py      # Correctness tests
│   └── benchmark_ring_attention.py # Performance benchmarks
├── scripts/
│   └── prepare_data.py    # OpenWebText tokenization
└── data/
    ├── train.bin          # Tokenized training data
    └── val.bin            # Tokenized validation data
```

## Installation

```bash
pip install torch>=2.2.0 numpy<2.0 transformers datasets fire
```

## Usage

### Run Correctness Tests

```bash
python tests/test_ring_attention.py
```

Tests include:
- Ring attention (causal) vs single-GPU reference
- Ring attention (non-causal) vs single-GPU reference
- Full LLaMA model with ring attention vs vanilla attention

### Run Performance Benchmarks

```bash
# Compare vanilla vs ring attention
python tests/benchmark_ring_attention.py --mode compare

# With bfloat16 precision
python tests/benchmark_ring_attention.py --mode compare --bf16

# Custom configuration
python tests/benchmark_ring_attention.py --mode compare \
    --dim 1024 --n-layers 8 --n-heads 16 --world-size 4
```

### Using the Model

```python
from model.llama import Transformer, ModelArgs

# Create model
args = ModelArgs(
    dim=512,
    n_layers=4,
    n_heads=8,
    n_kv_heads=2,  # GQA
    vocab_size=32000,
    max_seq_len=2048,
)
model = Transformer(args)

# Standard forward pass
logits = model(input_ids)

# Distributed forward with ring attention
logits = model(input_ids_local, use_ring_attention=True)
```

## Ring Attention: How It Works

1. **Sequence Split**: Global sequence is split across N GPUs
2. **Q Stays Local**: Each GPU keeps its Q chunk
3. **K,V Rotate**: K,V chunks rotate around the ring in N steps
4. **Online Softmax**: Numerically stable attention accumulation across chunks
5. **Communication Hiding**: Async send/recv overlaps with compute

### Causal Masking

```
Rank 0 (Q[0:256]):
  Step 0: Attend to K[0:256]   ← diagonal block (causal mask)
  Step 1: Attend to K[768:1024] ← future, fully masked
  Step 2: Attend to K[512:768]  ← future, fully masked
  Step 3: Attend to K[256:512]  ← future, fully masked

Rank 3 (Q[768:1024]):
  Step 0: Attend to K[768:1024] ← diagonal block (causal mask)
  Step 1: Attend to K[512:768]  ← past, no mask
  Step 2: Attend to K[256:512]  ← past, no mask
  Step 3: Attend to K[0:256]    ← past, no mask
```

## Benchmark Metrics

| Metric | Description |
|--------|-------------|
| Latency | Time per forward pass (ms) |
| Throughput | Tokens processed per second |
| MFU | Model FLOPs Utilization (achieved/peak TFLOPS) |
| TensorCore Util | % of CUDA time in matmul operations |
| Comm/Compute | Communication overhead ratio |
| Memory | Peak GPU memory usage |

Supported GPUs: A100, H100

## Performance Expectations

| Sequence Length | Vanilla | Ring Attention |
|-----------------|---------|----------------|
| Short (< 4K) | Faster (no comm overhead) | Slight overhead |
| Medium (4K-32K) | Limited by memory | Near-vanilla latency |
| Long (32K+) | OOM | Only option that works |

Ring attention achieves near-vanilla latency when communication is hidden behind compute (T_comm < T_compute).

## References

- [Ring Attention Paper](https://arxiv.org/abs/2310.01889) - Ring Attention with Blockwise Transformers for Near-Infinite Context
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - LLaMA: Open and Efficient Foundation Language Models
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Online softmax algorithm
