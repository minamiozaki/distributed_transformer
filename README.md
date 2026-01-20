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

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/distributed_transformer.git
cd distributed_transformer
pip install -r requirements.txt

# Verify correctness
python tests/test_ring_attention.py

# Run benchmark (CPU/Mac)
python tests/benchmark_ring_attention.py --mode compare

# Run benchmark (Multi-GPU with NCCL)
python tests/benchmark_ring_attention.py --mode compare --world-size 4
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

## RoPE Positioning in Distributed Mode

Each rank must use **global** position indices for RoPE, not local:

```
Global sequence: [token_0, token_1, ..., token_1023]
                      ↓
Split across 4 GPUs:
  Rank 0: tokens [0:256]   → freqs_cis[0:256]
  Rank 1: tokens [256:512] → freqs_cis[256:512]
  Rank 2: tokens [512:768] → freqs_cis[512:768]
  Rank 3: tokens [768:1024]→ freqs_cis[768:1024]
```

If rank 1 incorrectly used `freqs_cis[0:256]`, its tokens would have wrong positional info, breaking attention.

```python
# In Transformer.forward():
if use_ring_attention:
    rank = dist.get_rank(process_group)
    start_pos = rank * S  # Global position offset
    freqs_cis = self.freqs_cis[start_pos:start_pos + S]
else:
    freqs_cis = self.freqs_cis[:S]
```

## Online Softmax Algorithm

Ring attention uses online softmax to accumulate attention across K,V chunks without materializing the full attention matrix:

```python
# For each K,V chunk from the ring:
scores = Q @ K.T / sqrt(d)
block_max = scores.max()

# Update running max
new_max = max(running_max, block_max)

# Rescale previous output
scale = exp(running_max - new_max)
output = output * scale

# Add current block's contribution
weights = exp(scores - new_max)
output += weights @ V

# Update normalizer
normalizer = normalizer * scale + weights.sum()

# Final output
output = output / normalizer
```

This is numerically stable and memory efficient - we never store the full [S, S] attention matrix.

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

### MFU (Model FLOPs Utilization)

MFU measures how efficiently you use the GPU's compute capability:

```
MFU = Achieved TFLOPS / Peak TFLOPS
```

We use **actual profiled FLOPs** from PyTorch profiler (`with_flops=True`), not estimates:

```python
with torch.profiler.profile(with_flops=True) as prof:
    model(input_ids)

total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
achieved_tflops = total_flops / latency_sec / 1e12
mfu = achieved_tflops / peak_tflops
```

| MFU Range | Interpretation |
|-----------|----------------|
| 50%+ | Excellent |
| 30-50% | Good (typical for training) |
| 15-30% | Decent |
| <15% | Likely memory-bound |

### TensorCore Utilization

Percentage of GPU time spent in matmul operations (which use TensorCores):

```
TC% = matmul_cuda_time / total_cuda_time
```

Higher is better - means more time doing useful matrix math vs overhead.

### Comm/Compute Ratio

For ring attention, measures communication overhead:

```
Comm% = barrier_sync_time / compute_time
```

Lower is better. When `T_comm < T_compute`, communication is fully hidden.

## Distributed Backends

| Backend | Device | Use Case |
|---------|--------|----------|
| NCCL | CUDA GPUs | Production multi-GPU (fastest) |
| Gloo | CPU | Testing, Mac development |

```python
# Automatic selection in benchmark
backend = 'nccl' if torch.cuda.is_available() else 'gloo'
```

NCCL uses NVLink/PCIe for direct GPU-to-GPU communication. Gloo routes through CPU memory.

## Performance Expectations

| Sequence Length | Vanilla Attention | Ring Attention | Winner |
|-----------------|-------------------|----------------|--------|
| Short (< 4K) | ~1-5ms | ~2-8ms | Vanilla (no comm overhead) |
| Medium (4K-32K) | ~50-500ms | ~50-500ms | Tie (comm hidden) |
| Long (32K+) | OOM | Works | Ring (only option) |
| Very Long (128K+) | OOM | Works | Ring (enables new capabilities) |

**Key insight**: Ring attention's value isn't speed—it's enabling sequence lengths that would otherwise be impossible due to memory constraints.

** 1/19 Update on Basic Performance on 2 x H100 GPU server **
## Performance Benchmark                                                                   
                                                                                             
  Comparison of Ring Attention vs Vanilla Attention on 2x GPUs.                              
                                                                                             
  | Seq Len | Vanilla (ms) | Ring (ms) | Speedup | V-MFU | R-MFU | V-Mem (GB) | R-Mem (GB) | 
  TC% | Comm% |                                                                              
  |--------:|-------------:|----------:|--------:|------:|------:|-----------:|-----------:|-
  ---:|------:|                                                                              
  | 512 | 2.73 | 8.44 | 0.32x | 15.5% | 5.4% | 0.34 | 0.28 | 53% | 1.2% |                    
  | 1024 | 3.20 | 8.57 | 0.37x | 26.4% | 11.4% | 0.46 | 0.34 | 42% | 1.1% |                  
  | 2048 | 4.63 | 8.89 | 0.52x | 36.5% | 24.8% | 0.71 | 0.49 | 28% | 1.1% |                  
  | 4096 | 9.29 | 14.03 | 0.66x | 36.4% | 38.8% | 1.21 | 1.02 | 51% | 0.7% |                 
  | 8192 | 21.19 | 41.37 | 0.51x | 31.9% | 36.3% | 2.21 | 2.90 | 12% | 0.2% |                
  | 16384 | 57.83 | 145.17 | 0.40x | 23.4% | 32.0% | 4.21 | 9.83 | 6% | 0.1% |               
  | 32768 | 180.35 | 554.35 | 0.33x | 15.0% | 28.7% | 8.21 | 36.45 | 3% | 0.1% |             
                                                                                             
  **Legend:**                                                                                
  - **Speedup** > 1 means ring attention is faster                                           
  - **MFU** = Model FLOPs Utilization (higher is better)                                     
  - **V-Mem / R-Mem** = Peak memory for Vanilla / Ring (per GPU)                             
  - **TC%** = TensorCore utilization (matmul time / total CUDA time)                         
  - **Comm%** = Communication/Compute ratio for ring attention (lower is better) 

## Limitations & Future Work

**Current limitations:**
- Forward pass only (no backward pass / training)
- No FlashAttention integration (would further improve memory)
- Causal mask doesn't skip future chunks (potential optimization)

**Potential optimizations:**
- Skip attention computation for fully-masked future chunks
- Fuse K,V communication with attention computation
- Integrate with FlashAttention-2 for block-sparse attention

## References

- [Ring Attention Paper](https://arxiv.org/abs/2310.01889) - Ring Attention with Blockwise Transformers for Near-Infinite Context
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - LLaMA: Open and Efficient Foundation Language Models
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Online softmax algorithm
