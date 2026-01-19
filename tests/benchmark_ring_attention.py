"""
Benchmark Ring Attention vs Vanilla Attention performance.

Run with:
    # Single process (vanilla attention only)
    python tests/benchmark_ring_attention.py --mode vanilla

    # Multi-process (ring attention)
    torchrun --nproc_per_node=4 tests/benchmark_ring_attention.py --mode ring

    # Both modes for comparison (runs vanilla first, then spawns ring)
    python tests/benchmark_ring_attention.py --mode compare
"""
import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llama import Transformer, ModelArgs


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def benchmark_vanilla_attention(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: torch.device = None,
) -> dict:
    """Benchmark vanilla (single-GPU) attention."""
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Benchmarking Vanilla Attention")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"{'='*60}")

    # Create model
    torch.manual_seed(42)
    model = Transformer(args).to(device)
    model.eval()

    # Create input
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, use_ring_attention=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iters} iterations)...")
    latencies = []

    with torch.no_grad():
        for i in range(num_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids, use_ring_attention=False)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    # Compute stats
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput = (batch_size * seq_len) / (avg_latency / 1000)  # tokens/sec

    # Memory stats
    if device.type == 'cuda':
        mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    else:
        mem_allocated = 0
        mem_reserved = 0

    results = {
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'throughput_tokens_per_sec': throughput,
        'mem_allocated_gb': mem_allocated,
        'mem_reserved_gb': mem_reserved,
    }

    print(f"\nResults:")
    print(f"  Avg latency: {avg_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Throughput: {throughput:.0f} tokens/sec")
    if device.type == 'cuda':
        print(f"  Peak memory allocated: {mem_allocated:.2f} GB")
        print(f"  Peak memory reserved: {mem_reserved:.2f} GB")

    return results


def benchmark_ring_attention_worker(
    rank: int,
    world_size: int,
    args: ModelArgs,
    batch_size: int,
    seq_len_global: int,
    num_warmup: int,
    num_iters: int,
    results_queue: mp.Queue,
    backend: str = 'gloo',
):
    """Worker function for ring attention benchmark."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'

    # Use NCCL for CUDA, Gloo for CPU
    if torch.cuda.is_available() and backend == 'nccl':
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    try:
        seq_len_local = seq_len_global // world_size

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Benchmarking Ring Attention")
            print(f"{'='*60}")
            print(f"Device: {device}")
            print(f"World size: {world_size}")
            print(f"Model: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
            print(f"Batch size: {batch_size}")
            print(f"Sequence length: {seq_len_global} global, {seq_len_local} per rank")
            print(f"{'='*60}")

        # Create model (same seed on all ranks)
        torch.manual_seed(42)
        model = Transformer(args).to(device)
        model.eval()

        # Create input (same on all ranks, then split)
        torch.manual_seed(123)
        input_ids_full = torch.randint(0, args.vocab_size, (batch_size, seq_len_global))
        start_idx = rank * seq_len_local
        end_idx = start_idx + seq_len_local
        input_ids_local = input_ids_full[:, start_idx:end_idx].to(device).contiguous()

        # Warmup
        if rank == 0:
            print(f"Warming up ({num_warmup} iterations)...")

        dist.barrier()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_ids_local, use_ring_attention=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        if rank == 0:
            print(f"Benchmarking ({num_iters} iterations)...")

        latencies = []
        dist.barrier()

        with torch.no_grad():
            for i in range(num_iters):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dist.barrier()

                start = time.perf_counter()
                _ = model(input_ids_local, use_ring_attention=True)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dist.barrier()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        # Compute stats (only on rank 0)
        if rank == 0:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            throughput = (batch_size * seq_len_global) / (avg_latency / 1000)

            if device.type == 'cuda':
                mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            else:
                mem_allocated = 0
                mem_reserved = 0

            results = {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'throughput_tokens_per_sec': throughput,
                'mem_allocated_gb': mem_allocated,
                'mem_reserved_gb': mem_reserved,
                'world_size': world_size,
            }

            print(f"\nResults:")
            print(f"  Avg latency: {avg_latency:.2f} ms")
            print(f"  Min latency: {min_latency:.2f} ms")
            print(f"  Max latency: {max_latency:.2f} ms")
            print(f"  Throughput: {throughput:.0f} tokens/sec")
            if device.type == 'cuda':
                print(f"  Peak memory allocated (per GPU): {mem_allocated:.2f} GB")
                print(f"  Peak memory reserved (per GPU): {mem_reserved:.2f} GB")

            results_queue.put(results)

    finally:
        dist.destroy_process_group()


def benchmark_ring_attention(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    world_size: int = 4,
    num_warmup: int = 5,
    num_iters: int = 20,
    backend: str = 'gloo',
) -> dict:
    """Benchmark ring attention with multiple processes."""
    mp.set_start_method('spawn', force=True)

    results_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_ring_attention_worker,
            args=(rank, world_size, args, batch_size, seq_len,
                  num_warmup, num_iters, results_queue, backend)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if not results_queue.empty():
        return results_queue.get()
    return {}


def run_comparison(
    seq_lengths: list = None,
    world_size: int = 4,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iters: int = 20,
):
    """Run comparison between vanilla and ring attention."""
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048]

    # Model config - smaller for benchmarking
    args = ModelArgs(
        dim=512,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32000,
        max_seq_len=max(seq_lengths) + 128,
        multiple_of=64,
    )

    device = get_device()
    backend = 'nccl' if device.type == 'cuda' else 'gloo'

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: Vanilla vs Ring Attention")
    print("="*70)
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"World size (for ring): {world_size}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"heads={args.n_heads}, kv_heads={args.n_kv_heads}")
    print("="*70)

    all_results = []

    for seq_len in seq_lengths:
        print(f"\n{'#'*70}")
        print(f"# Sequence Length: {seq_len}")
        print(f"{'#'*70}")

        # Vanilla attention
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        vanilla_results = benchmark_vanilla_attention(
            args, batch_size, seq_len, num_warmup, num_iters, device
        )

        # Ring attention
        ring_results = benchmark_ring_attention(
            args, batch_size, seq_len, world_size, num_warmup, num_iters, backend
        )

        # Compare
        if vanilla_results and ring_results:
            speedup = vanilla_results['avg_latency_ms'] / ring_results['avg_latency_ms']
            mem_ratio = (vanilla_results['mem_allocated_gb'] /
                        ring_results['mem_allocated_gb']) if ring_results['mem_allocated_gb'] > 0 else 0

            all_results.append({
                'seq_len': seq_len,
                'vanilla_latency': vanilla_results['avg_latency_ms'],
                'ring_latency': ring_results['avg_latency_ms'],
                'speedup': speedup,
                'vanilla_mem': vanilla_results['mem_allocated_gb'],
                'ring_mem_per_gpu': ring_results['mem_allocated_gb'],
            })

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq Len':>10} | {'Vanilla (ms)':>12} | {'Ring (ms)':>12} | {'Speedup':>8} | {'Mem Ratio':>10}")
    print("-"*70)

    for r in all_results:
        mem_str = f"{r['vanilla_mem']:.2f}/{r['ring_mem_per_gpu']:.2f}" if r['ring_mem_per_gpu'] > 0 else "N/A"
        print(f"{r['seq_len']:>10} | {r['vanilla_latency']:>12.2f} | {r['ring_latency']:>12.2f} | "
              f"{r['speedup']:>8.2f}x | {mem_str:>10}")

    print("="*70)
    print("Note: Ring attention memory is per-GPU. Speedup > 1 means ring is faster.")
    print("="*70)

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ring Attention')
    parser.add_argument('--mode', choices=['vanilla', 'ring', 'compare'], default='compare',
                       help='Benchmark mode')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--world-size', type=int, default=4, help='World size for ring attention')
    parser.add_argument('--num-warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--num-iters', type=int, default=20, help='Benchmark iterations')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-kv-heads', type=int, default=2, help='Number of KV heads (GQA)')

    args = parser.parse_args()

    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=32000,
        max_seq_len=args.seq_len + 128,
        multiple_of=64,
    )

    if args.mode == 'vanilla':
        benchmark_vanilla_attention(
            model_args, args.batch_size, args.seq_len,
            args.num_warmup, args.num_iters
        )

    elif args.mode == 'ring':
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        benchmark_ring_attention(
            model_args, args.batch_size, args.seq_len,
            args.world_size, args.num_warmup, args.num_iters, backend
        )

    elif args.mode == 'compare':
        run_comparison(
            seq_lengths=[512, 1024, 2048],
            world_size=args.world_size,
            batch_size=args.batch_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )


if __name__ == '__main__':
    main()
