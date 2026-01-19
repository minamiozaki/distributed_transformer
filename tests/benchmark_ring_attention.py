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


# GPU specs for MFU calculation (peak TFLOPS)
GPU_SPECS = {
    # GPU name: (FP32 TFLOPS, FP16/BF16 TensorCore TFLOPS)
    'A100': (19.5, 312),
    'A100-80GB': (19.5, 312),
    'H100': (67, 989),
    'H100-SXM': (67, 989),
    'H100-PCIe': (51, 756),
    'default': (19.5, 312),  # Assume A100 if unknown
}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_gpu_name():
    """Get GPU name for specs lookup."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        # Match against known GPUs
        for key in GPU_SPECS:
            if key in name:
                return key
    return 'default'


def get_gpu_peak_tflops(dtype=torch.float32):
    """Get theoretical peak TFLOPS for current GPU."""
    gpu_name = get_gpu_name()
    fp32_tflops, tensor_tflops = GPU_SPECS.get(gpu_name, GPU_SPECS['default'])

    if dtype in (torch.float16, torch.bfloat16):
        return tensor_tflops
    return fp32_tflops


def count_parameters(model):
    """Count total model parameters."""
    return sum(p.numel() for p in model.parameters())


def profile_flops_and_metrics(model, input_ids, num_iters=5, use_ring_attention=False):
    """
    Profile model to get actual FLOPs and TensorCore utilization.
    Uses PyTorch profiler with with_flops=True for accurate measurement.

    Returns profiler stats if CUDA is available, otherwise estimates.
    """
    device = input_ids.device

    if device.type == 'cuda':
        try:
            from torch.profiler import profile, ProfilerActivity

            # Profile multiple iterations for stability
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    for _ in range(num_iters):
                        _ = model(input_ids, use_ring_attention=use_ring_attention)
                        torch.cuda.synchronize()

            # Extract stats
            key_averages = prof.key_averages()

            # Total CUDA time
            total_cuda_time_us = sum(e.cuda_time_total for e in key_averages)

            # Find matmul/GEMM operations (TensorCore candidates)
            matmul_time_us = 0
            for e in key_averages:
                if any(op in e.key.lower() for op in ['matmul', 'gemm', 'mm', 'bmm', 'linear', 'addmm']):
                    matmul_time_us += e.cuda_time_total

            # TensorCore utilization estimate (matmul time / total CUDA time)
            tensorcore_util = matmul_time_us / total_cuda_time_us if total_cuda_time_us > 0 else 0

            # Get total FLOPs from profiler (this is the actual measured FLOPs)
            total_flops = sum(e.flops for e in key_averages if e.flops > 0)

            # FLOPs per iteration
            flops_per_iter = total_flops / num_iters

            return {
                'total_flops': flops_per_iter,
                'tensorcore_util': tensorcore_util,
                'matmul_time_us': matmul_time_us / num_iters,
                'total_cuda_time_us': total_cuda_time_us / num_iters,
                'profiler_available': True,
            }
        except Exception as e:
            print(f"  Profiling failed: {e}")
            return {
                'total_flops': 0,
                'tensorcore_util': 0,
                'profiler_available': False,
            }
    else:
        # No CUDA, can't profile accurately
        return {
            'total_flops': 0,
            'tensorcore_util': 0,
            'profiler_available': False,
        }


def compute_mfu(total_flops, latency_sec, dtype=torch.float32):
    """
    Compute Model FLOPs Utilization (MFU) using actual profiled FLOPs.

    MFU = Achieved FLOPS / Theoretical Peak FLOPS

    Args:
        total_flops: Actual FLOPs from profiler for one forward pass
        latency_sec: Time for one forward pass in seconds
        dtype: Data type for peak TFLOPS lookup
    """
    if total_flops == 0 or latency_sec == 0:
        return {
            'mfu': 0,
            'achieved_tflops': 0,
            'peak_tflops': get_gpu_peak_tflops(dtype),
            'total_flops': total_flops,
        }

    # Achieved TFLOPS
    achieved_tflops = total_flops / latency_sec / 1e12

    # Peak TFLOPS
    peak_tflops = get_gpu_peak_tflops(dtype)

    mfu = achieved_tflops / peak_tflops if peak_tflops > 0 else 0

    return {
        'mfu': mfu,
        'achieved_tflops': achieved_tflops,
        'peak_tflops': peak_tflops,
        'total_flops': total_flops,
    }


def benchmark_vanilla_attention(
    args: ModelArgs,
    batch_size: int,
    seq_len: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Benchmark vanilla (single-GPU) attention."""
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"Benchmarking Vanilla Attention")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"{'='*60}")

    # Create model
    torch.manual_seed(42)
    model = Transformer(args).to(device)
    if dtype == torch.bfloat16 and device.type == 'cuda':
        model = model.to(dtype=dtype)
    model.eval()

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create input
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, use_ring_attention=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Profile to get actual FLOPs and TensorCore utilization
    print("Profiling for FLOPs and TensorCore utilization...")
    profile_stats = profile_flops_and_metrics(model, input_ids, num_iters=3, use_ring_attention=False)

    # Reset memory stats after warmup/profiling
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

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

    # MFU using actual profiled FLOPs
    mfu_stats = compute_mfu(profile_stats['total_flops'], avg_latency / 1000, dtype)

    results = {
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'throughput_tokens_per_sec': throughput,
        'mem_allocated_gb': mem_allocated,
        'mem_reserved_gb': mem_reserved,
        'num_params': num_params,
        'tensorcore_util': profile_stats.get('tensorcore_util', 0),
        **mfu_stats,
    }

    print(f"\nResults:")
    print(f"  Avg latency: {avg_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")
    if profile_stats['profiler_available']:
        print(f"  Profiled FLOPs: {profile_stats['total_flops']:.2e}")
        print(f"  MFU: {mfu_stats['mfu']:.1%}")
        print(f"  Achieved TFLOPS: {mfu_stats['achieved_tflops']:.2f}")
        print(f"  TensorCore utilization: {profile_stats['tensorcore_util']:.1%}")
    else:
        print(f"  MFU: N/A (profiler not available)")
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
    dtype: torch.dtype = torch.float32,
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
            if device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"World size: {world_size}")
            print(f"Model: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}")
            print(f"Batch size: {batch_size}")
            print(f"Sequence length: {seq_len_global} global, {seq_len_local} per rank")
            print(f"{'='*60}")

        # Create model (same seed on all ranks)
        torch.manual_seed(42)
        model = Transformer(args).to(device)
        if dtype == torch.bfloat16 and device.type == 'cuda':
            model = model.to(dtype=dtype)
        model.eval()

        num_params = count_parameters(model)
        if rank == 0:
            print(f"Model parameters: {num_params / 1e6:.2f}M")

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

        # Profile FLOPs on rank 0 only (to avoid distributed profiling complexity)
        # Note: For ring attention, we profile the local computation
        # Total system FLOPs = local FLOPs * world_size (approximately)
        profile_stats = {'total_flops': 0, 'tensorcore_util': 0, 'profiler_available': False}
        if rank == 0 and device.type == 'cuda':
            print("Profiling for FLOPs (rank 0 only)...")
            # Profile without ring attention to get per-GPU FLOPs baseline
            # Then multiply by world_size for total system FLOPs
            profile_stats = profile_flops_and_metrics(
                model, input_ids_local, num_iters=3, use_ring_attention=True
            )
            # Scale FLOPs to represent total system work
            # Each GPU processes full attention over all K,V chunks
            profile_stats['total_flops'] = profile_stats['total_flops'] * world_size

        dist.barrier()

        # Reset memory stats after warmup/profiling
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # Benchmark with comm/compute breakdown
        if rank == 0:
            print(f"Benchmarking ({num_iters} iterations)...")

        latencies = []
        compute_times = []
        comm_times = []

        dist.barrier()

        with torch.no_grad():
            for i in range(num_iters):
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                # Measure barrier (communication sync) time
                comm_start = time.perf_counter()
                dist.barrier()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                comm_end = time.perf_counter()

                # Measure total iteration time
                start = time.perf_counter()
                _ = model(input_ids_local, use_ring_attention=True)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dist.barrier()

                end = time.perf_counter()

                total_time = (end - start) * 1000  # ms
                barrier_time = (comm_end - comm_start) * 1000  # ms

                latencies.append(total_time)
                comm_times.append(barrier_time)
                compute_times.append(total_time - barrier_time)

        # Compute stats (only on rank 0)
        if rank == 0:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            throughput = (batch_size * seq_len_global) / (avg_latency / 1000)

            avg_compute = sum(compute_times) / len(compute_times)
            avg_comm = sum(comm_times) / len(comm_times)

            # Comm/Compute ratio
            comm_compute_ratio = avg_comm / avg_compute if avg_compute > 0 else 0

            if device.type == 'cuda':
                mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            else:
                mem_allocated = 0
                mem_reserved = 0

            # MFU using actual profiled FLOPs
            mfu_stats = compute_mfu(profile_stats['total_flops'], avg_latency / 1000, dtype)

            results = {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'throughput_tokens_per_sec': throughput,
                'mem_allocated_gb': mem_allocated,
                'mem_reserved_gb': mem_reserved,
                'world_size': world_size,
                'num_params': num_params,
                'avg_compute_ms': avg_compute,
                'avg_comm_ms': avg_comm,
                'comm_compute_ratio': comm_compute_ratio,
                'tensorcore_util': profile_stats.get('tensorcore_util', 0),
                **mfu_stats,
            }

            print(f"\nResults:")
            print(f"  Avg latency: {avg_latency:.2f} ms")
            print(f"  Min latency: {min_latency:.2f} ms")
            print(f"  Max latency: {max_latency:.2f} ms")
            print(f"  Throughput: {throughput:,.0f} tokens/sec")
            if profile_stats['profiler_available']:
                print(f"  Profiled FLOPs (system): {profile_stats['total_flops']:.2e}")
                print(f"  MFU (system): {mfu_stats['mfu']:.1%}")
                print(f"  Achieved TFLOPS (system): {mfu_stats['achieved_tflops']:.2f}")
                print(f"  TensorCore utilization: {profile_stats['tensorcore_util']:.1%}")
            else:
                print(f"  MFU: N/A (profiler not available)")
            print(f"  Avg compute time: {avg_compute:.2f} ms")
            print(f"  Avg comm overhead: {avg_comm:.2f} ms")
            print(f"  Comm/Compute ratio: {comm_compute_ratio:.2%}")
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
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Benchmark ring attention with multiple processes."""
    mp.set_start_method('spawn', force=True)

    results_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_ring_attention_worker,
            args=(rank, world_size, args, batch_size, seq_len,
                  num_warmup, num_iters, results_queue, backend, dtype)
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
    dtype: torch.dtype = torch.float32,
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

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Vanilla vs Ring Attention")
    print("="*80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Peak TFLOPS: {get_gpu_peak_tflops(dtype):.1f}")
    print(f"Backend: {backend}")
    print(f"World size (for ring): {world_size}")
    print(f"Model: dim={args.dim}, layers={args.n_layers}, "
          f"heads={args.n_heads}, kv_heads={args.n_kv_heads}")
    print(f"Dtype: {dtype}")
    print("="*80)

    all_results = []

    for seq_len in seq_lengths:
        print(f"\n{'#'*80}")
        print(f"# Sequence Length: {seq_len}")
        print(f"{'#'*80}")

        # Vanilla attention
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        vanilla_results = benchmark_vanilla_attention(
            args, batch_size, seq_len, num_warmup, num_iters, device, dtype
        )

        # Ring attention
        ring_results = benchmark_ring_attention(
            args, batch_size, seq_len, world_size, num_warmup, num_iters, backend, dtype
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
                'vanilla_mfu': vanilla_results.get('mfu', 0),
                'ring_mfu': ring_results.get('mfu', 0),
                'vanilla_throughput': vanilla_results['throughput_tokens_per_sec'],
                'ring_throughput': ring_results['throughput_tokens_per_sec'],
                'ring_comm_ratio': ring_results.get('comm_compute_ratio', 0),
                'vanilla_tc_util': vanilla_results.get('tensorcore_util', 0),
                'ring_tc_util': ring_results.get('tensorcore_util', 0),
            })

    # Summary table
    print("\n" + "="*110)
    print("SUMMARY")
    print("="*110)

    # Header
    print(f"{'Seq Len':>8} | {'Vanilla':>10} | {'Ring':>10} | {'Speedup':>8} | "
          f"{'V-MFU':>6} | {'R-MFU':>6} | {'V-Mem':>6} | {'R-Mem':>6} | {'TC%':>5} | {'Comm%':>6}")
    print(f"{'':>8} | {'(ms)':>10} | {'(ms)':>10} | {'':>8} | "
          f"{'':>6} | {'':>6} | {'(GB)':>6} | {'(GB)':>6} | {'':>5} | {'':>6}")
    print("-"*110)

    for r in all_results:
        print(f"{r['seq_len']:>8} | {r['vanilla_latency']:>10.2f} | {r['ring_latency']:>10.2f} | "
              f"{r['speedup']:>7.2f}x | {r['vanilla_mfu']:>5.1%} | {r['ring_mfu']:>5.1%} | "
              f"{r['vanilla_mem']:>6.2f} | {r['ring_mem_per_gpu']:>6.2f} | "
              f"{r['vanilla_tc_util']:>4.0%} | {r['ring_comm_ratio']:>5.1%}")

    print("="*110)
    print("Legend:")
    print("  Speedup > 1 means ring is faster")
    print("  MFU = Model FLOPs Utilization (higher is better, uses profiled FLOPs)")
    print("  V-Mem = Vanilla peak memory, R-Mem = Ring peak memory per GPU")
    print("  TC% = TensorCore utilization (matmul time / total CUDA time)")
    print("  Comm% = Communication/Compute ratio for ring attention (lower is better)")
    print("="*110)

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
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision')

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32

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
            args.num_warmup, args.num_iters, dtype=dtype
        )

    elif args.mode == 'ring':
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        benchmark_ring_attention(
            model_args, args.batch_size, args.seq_len,
            args.world_size, args.num_warmup, args.num_iters, backend, dtype
        )

    elif args.mode == 'compare':
        run_comparison(
            seq_lengths=[512, 1024, 2048],
            world_size=args.world_size,
            batch_size=args.batch_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            dtype=dtype,
        )


if __name__ == '__main__':
    main()
