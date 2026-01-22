"""Minimal debug script for ring attention hang."""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ring_attention import ring_attention_forward_flash, ring_attention_forward_eager, HAS_FLASH_ATTN
from model.llama import Transformer, ModelArgs


def log(rank, msg):
    """Write to file to ensure output is captured."""
    with open(f"/tmp/ring_debug_rank{rank}.log", "a") as f:
        f.write(f"{msg}\n")
        f.flush()


def worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    log(rank, f"Worker {rank} starting")

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    log(rank, f"Worker {rank} initializing process group")
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    log(rank, f"Worker {rank} process group initialized")

    try:
        # Use same model config as benchmark
        args = ModelArgs(
            dim=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=32000,
            max_seq_len=1024,
            multiple_of=64,
        )

        torch.manual_seed(42)
        model = Transformer(args).to(device).to(torch.bfloat16)
        model.eval()
        log(rank, f"Worker {rank} model created")

        # Create input
        seq_len_global = 512
        seq_len_local = seq_len_global // world_size
        torch.manual_seed(123)
        input_ids_full = torch.randint(0, args.vocab_size, (1, seq_len_global))
        start_idx = rank * seq_len_local
        end_idx = start_idx + seq_len_local
        input_ids_local = input_ids_full[:, start_idx:end_idx].to(device).contiguous()
        log(rank, f"Worker {rank} input created, shape={input_ids_local.shape}")

        dist.barrier()
        log(rank, f"Worker {rank} passed barrier, starting forward")

        # Try multiple forward passes like the benchmark
        with torch.no_grad():
            for i in range(3):
                log(rank, f"Worker {rank} forward pass {i}")
                out = model(input_ids_local, use_ring_attention=True)
                torch.cuda.synchronize()
                log(rank, f"Worker {rank} forward pass {i} done, shape={out.shape}")

        dist.barrier()
        log(rank, f"Worker {rank} DONE")

    except Exception as e:
        log(rank, f"Worker {rank} ERROR: {e}")
        import traceback
        log(rank, traceback.format_exc())
    finally:
        dist.destroy_process_group()


def main():
    # Clear old logs
    for i in range(2):
        try:
            os.remove(f"/tmp/ring_debug_rank{i}.log")
        except FileNotFoundError:
            pass

    world_size = 2
    mp.set_start_method('spawn', force=True)

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            print(f"Process {p.pid} timed out, terminating")
            p.terminate()
            p.join()

    # Print logs
    print("\n=== Debug logs ===")
    for i in range(world_size):
        print(f"\n--- Rank {i} ---")
        try:
            with open(f"/tmp/ring_debug_rank{i}.log") as f:
                print(f.read())
        except FileNotFoundError:
            print("(no log file)")


if __name__ == '__main__':
    main()
