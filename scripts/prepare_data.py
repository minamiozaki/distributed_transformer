# scripts/prepare_data_v2.py
import os
import sys
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# --- CONFIG ---
IS_MAC = sys.platform == "darwin"

# Mac: Stop after 20MB. Server: Go forever.
MAX_TOKENS = 10 * 1024 * 1024 if IS_MAC else float('inf')

# Flush to disk every 100 million tokens (~200MB file size) to save RAM
FLUSH_SIZE = 100 * 1000 * 1000 

def write_batch(filename, buffer):
    if not buffer: return
    dtype = np.uint16
    arr = np.array(buffer, dtype=dtype)
    # Append to file ('ab' mode)
    with open(filename, 'ab') as f:
        f.write(arr.tobytes())

def process_stream():
    enc = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    
    # Paths
    os.makedirs('data', exist_ok=True)
    train_path = os.path.join('data', 'train.bin')
    val_path = os.path.join('data', 'val.bin')
    
    # Clear existing files so we don't append to old runs
    if os.path.exists(train_path): os.remove(train_path)
    if os.path.exists(val_path): os.remove(val_path)

    # Buffers
    train_buffer = []
    val_buffer = []
    total_count = 0
    
    print(f"Streaming data... (Limit: {MAX_TOKENS} tokens)")
    
    for example in tqdm(dataset):
        ids = enc.encode(example['text'])
        
        # 90/10 Split
        if random.random() < 0.9:
            train_buffer.extend(ids)
        else:
            val_buffer.extend(ids)
            
        total_count += len(ids)
        
        # Flush if buffers get big
        if len(train_buffer) > FLUSH_SIZE:
            write_batch(train_path, train_buffer)
            train_buffer = [] # Clear RAM
            
        if len(val_buffer) > FLUSH_SIZE:
            write_batch(val_path, val_buffer)
            val_buffer = []
            
        # Stop limit
        if total_count >= MAX_TOKENS:
            print(f"Hit limit ({total_count}). Stopping.")
            break

    # Final Flush
    write_batch(train_path, train_buffer)
    write_batch(val_path, val_buffer)

if __name__ == '__main__':
    process_stream()
    print("Done. train.bin and val.bin are ready.")