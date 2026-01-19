"""
Test Ring Attention correctness by comparing against single-GPU attention.

Run with:
    python tests/test_ring_attention.py

This spawns multiple processes using torch.multiprocessing to simulate
distributed training with the Gloo backend (works on CPU/Mac).
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ring_attention import ring_attention_forward, repeat_kv
from model.llama import Transformer, ModelArgs


def reference_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n_rep: int,
    is_causal: bool = True
) -> torch.Tensor:
    """
    Standard single-GPU attention for reference.

    Args:
        Q: [B, n_heads, S, head_dim]
        K: [B, n_kv_heads, S, head_dim]
        V: [B, n_kv_heads, S, head_dim]
        n_rep: GQA repetition factor
        is_causal: Whether to apply causal mask

    Returns:
        [B, n_heads, S, head_dim]
    """
    K = repeat_kv(K, n_rep)
    V = repeat_kv(V, n_rep)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def run_forward_test(rank: int, world_size: int, results_queue: mp.Queue):
    """
    Test function run on each process.

    Compares ring attention output against single-GPU reference.
    """
    # Initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

    try:
        # Test parameters
        B = 1           # batch size
        n_heads = 8     # query heads
        n_kv_heads = 2  # key/value heads (GQA)
        S_global = 2048 # total sequence length
        head_dim = 64   # dimension per head
        n_rep = n_heads // n_kv_heads
        S_local = S_global // world_size

        # Set seed for reproducibility (same on all ranks)
        torch.manual_seed(42)

        # Create full tensors (same on all ranks due to same seed)
        Q_full = torch.randn(B, n_heads, S_global, head_dim)
        K_full = torch.randn(B, n_kv_heads, S_global, head_dim)
        V_full = torch.randn(B, n_kv_heads, S_global, head_dim)

        # Compute reference on all ranks (for comparison)
        with torch.no_grad():
            O_ref = reference_attention(Q_full, K_full, V_full, n_rep, is_causal=True)

        # Split tensors for this rank
        start = rank * S_local
        end = start + S_local

        Q_local = Q_full[:, :, start:end, :].contiguous()
        K_local = K_full[:, :, start:end, :].contiguous()
        V_local = V_full[:, :, start:end, :].contiguous()

        # Run ring attention
        with torch.no_grad():
            O_local = ring_attention_forward(
                Q_local, K_local, V_local,
                process_group=None,  # Use default WORLD group
                n_rep=n_rep,
                is_causal=True
            )

        # Gather outputs from all ranks
        O_gathered = [torch.empty_like(O_local) for _ in range(world_size)]
        dist.all_gather(O_gathered, O_local)
        O_ring = torch.cat(O_gathered, dim=2)

        # Compare against reference
        O_ref_local = O_ref[:, :, start:end, :]
        local_diff = (O_local - O_ref_local).abs().max().item()

        # Only rank 0 checks full output
        if rank == 0:
            full_diff = (O_ring - O_ref).abs().max().item()
            mean_diff = (O_ring - O_ref).abs().mean().item()

            results_queue.put({
                'status': 'success',
                'full_max_diff': full_diff,
                'full_mean_diff': mean_diff,
                'local_max_diff': local_diff,
            })

            print(f"\n{'='*50}")
            print(f"Ring Attention Forward Test Results")
            print(f"{'='*50}")
            print(f"World size: {world_size}")
            print(f"Sequence: {S_global} global, {S_local} per rank")
            print(f"Heads: {n_heads} query, {n_kv_heads} KV (GQA rep={n_rep})")
            print(f"{'='*50}")
            print(f"Max difference: {full_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
            print(f"{'='*50}")

            if full_diff < 1e-5:
                print("PASSED: Ring attention matches reference!")
            else:
                print(f"FAILED: Difference {full_diff:.2e} exceeds threshold 1e-5")

    except Exception as e:
        if rank == 0:
            results_queue.put({
                'status': 'error',
                'error': str(e)
            })
        raise

    finally:
        dist.destroy_process_group()


def test_ring_attention_forward():
    """Main test entry point."""
    world_size = 4

    # Use spawn method for multiprocessing
    mp.set_start_method('spawn', force=True)

    results_queue = mp.Queue()

    # Spawn processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_forward_test,
            args=(rank, world_size, results_queue)
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    # Check results
    if not results_queue.empty():
        result = results_queue.get()
        if result['status'] == 'error':
            raise RuntimeError(f"Test failed: {result['error']}")
        elif result['full_max_diff'] >= 1e-5:
            raise AssertionError(
                f"Ring attention output differs from reference by {result['full_max_diff']:.2e}"
            )

    print("\nAll tests passed!")


def run_non_causal_test(rank: int, world_size: int, queue: mp.Queue):
    """Test function for non-causal attention (must be at module level for pickling)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    try:
        torch.manual_seed(123)

        B, n_heads, n_kv_heads, S_global, head_dim = 1, 4, 2, 32, 16
        n_rep = n_heads // n_kv_heads
        S_local = S_global // world_size

        Q_full = torch.randn(B, n_heads, S_global, head_dim)
        K_full = torch.randn(B, n_kv_heads, S_global, head_dim)
        V_full = torch.randn(B, n_kv_heads, S_global, head_dim)

        with torch.no_grad():
            O_ref = reference_attention(Q_full, K_full, V_full, n_rep, is_causal=False)

        start = rank * S_local
        end = start + S_local

        Q_local = Q_full[:, :, start:end, :].contiguous()
        K_local = K_full[:, :, start:end, :].contiguous()
        V_local = V_full[:, :, start:end, :].contiguous()

        with torch.no_grad():
            O_local = ring_attention_forward(
                Q_local, K_local, V_local,
                n_rep=n_rep,
                is_causal=False
            )

        O_gathered = [torch.empty_like(O_local) for _ in range(world_size)]
        dist.all_gather(O_gathered, O_local)
        O_ring = torch.cat(O_gathered, dim=2)

        if rank == 0:
            diff = (O_ring - O_ref).abs().max().item()
            queue.put({'diff': diff})
            print(f"Non-causal test - max diff: {diff:.2e}")

    finally:
        dist.destroy_process_group()


def test_non_causal():
    """Test non-causal (bidirectional) attention."""
    world_size = 2

    queue = mp.Queue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_non_causal_test, args=(rank, world_size, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result = queue.get()
    assert result['diff'] < 1e-5, f"Non-causal test failed: diff={result['diff']:.2e}"
    print("Non-causal test passed!")


# ============================================================================
# Full Model Tests with Ring Attention
# ============================================================================

def run_full_model_test(rank: int, world_size: int, results_queue: mp.Queue):
    """
    Test full LLaMA model with ring attention against single-GPU reference.

    Each rank:
    1. Creates the same model (same seed)
    2. Gets its local chunk of input tokens
    3. Runs forward pass with ring attention
    4. Compares against reference single-GPU forward pass
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )

    try:
        # Small model config for testing
        args = ModelArgs(
            dim=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,  # GQA: 4x repetition
            vocab_size=1000,
            max_seq_len=512,
            multiple_of=64,
        )

        # Test parameters
        B = 1
        S_global = 256  # Total sequence length
        S_local = S_global // world_size

        # Create model with same seed on all ranks
        torch.manual_seed(42)
        model = Transformer(args)
        model.eval()

        # Create input tokens (same on all ranks)
        torch.manual_seed(123)
        input_ids_full = torch.randint(0, args.vocab_size, (B, S_global))

        # Reference: single-GPU forward pass with full sequence
        with torch.no_grad():
            logits_ref = model(input_ids_full, use_ring_attention=False)

        # Split input for this rank
        start = rank * S_local
        end = start + S_local
        input_ids_local = input_ids_full[:, start:end].contiguous()

        # Distributed: ring attention forward pass with local chunk
        with torch.no_grad():
            logits_local = model(
                input_ids_local,
                use_ring_attention=True,
                process_group=None,  # Use default WORLD group
            )

        # Gather outputs from all ranks
        logits_gathered = [torch.empty_like(logits_local) for _ in range(world_size)]
        dist.all_gather(logits_gathered, logits_local)
        logits_ring = torch.cat(logits_gathered, dim=1)

        # Compare
        logits_ref_local = logits_ref[:, start:end, :]
        local_diff = (logits_local - logits_ref_local).abs().max().item()

        if rank == 0:
            full_diff = (logits_ring - logits_ref).abs().max().item()
            mean_diff = (logits_ring - logits_ref).abs().mean().item()

            results_queue.put({
                'status': 'success',
                'full_max_diff': full_diff,
                'full_mean_diff': mean_diff,
                'local_max_diff': local_diff,
            })

            print(f"\n{'='*60}")
            print(f"Full LLaMA Model with Ring Attention Test Results")
            print(f"{'='*60}")
            print(f"Model: dim={args.dim}, layers={args.n_layers}, "
                  f"heads={args.n_heads}, kv_heads={args.n_kv_heads}")
            print(f"World size: {world_size}")
            print(f"Sequence: {S_global} global, {S_local} per rank")
            print(f"{'='*60}")
            print(f"Max difference: {full_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
            print(f"{'='*60}")

            if full_diff < 1e-4:
                print("PASSED: Ring attention model matches reference!")
            else:
                print(f"FAILED: Difference {full_diff:.2e} exceeds threshold 1e-4")

    except Exception as e:
        if rank == 0:
            results_queue.put({
                'status': 'error',
                'error': str(e)
            })
        raise

    finally:
        dist.destroy_process_group()


def test_full_model_ring_attention():
    """Test full LLaMA model with ring attention."""
    world_size = 4

    mp.set_start_method('spawn', force=True)

    results_queue = mp.Queue()

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_full_model_test,
            args=(rank, world_size, results_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if not results_queue.empty():
        result = results_queue.get()
        if result['status'] == 'error':
            raise RuntimeError(f"Test failed: {result['error']}")
        elif result['full_max_diff'] >= 1e-4:
            raise AssertionError(
                f"Model output differs from reference by {result['full_max_diff']:.2e}"
            )

    print("\nFull model test passed!")


if __name__ == '__main__':
    print("Testing Ring Attention (Causal)...")
    test_ring_attention_forward()

    print("\nTesting Ring Attention (Non-Causal)...")
    test_non_causal()

    print("\n" + "="*60)
    print("Testing Full LLaMA Model with Ring Attention...")
    print("="*60)
    test_full_model_ring_attention()
