import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096            # Llama 3 standard hidden dim
    n_layers: int = 32         # Depth
    n_heads: int = 32          # Attention Heads
    n_kv_heads: int = 8        # Grouped Query Attention (GQA)
    vocab_size: int = 50257    # GPT-2 vocab (matches your tokenizer)
    multiple_of: int = 1024    # For SwiGLU hidden layer sizing
    max_seq_len: int = 2048    # Will be overridden in training
    rope_theta: float = 500000.0

# --- 1. Rotary Positional Embeddings (RoPE) ---
# This is the "Complex Number" implementation used by Meta.
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Create complex numbers: cos(th) + i*sin(th)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape keys/queries to complex: [B, S, H, D/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs for broadcasting: [1, S, 1, D/2]
    ndim = xq_.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    # Rotate via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- 2. Building Blocks ---
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Ensure hidden_dim is a multiple of 'multiple_of' (usually 256 or 1024)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False) # Gate
        self.w2 = nn.Linear(args.dim, hidden_dim, bias=False) # Value
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False) # Output

    def forward(self, x):
        # SwiGLU: (Swish(Gate) * Value) -> Output
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# --- ADD THIS HELPER FUNCTION ---
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands the KV heads to match the Query heads.
    Input: [B, n_kv_heads, S, D]
    Output: [B, n_kv_heads * n_rep, S, D]
    """
    B, n_kv_heads, S, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, S, head_dim)
        .reshape(B, n_kv_heads * n_rep, S, head_dim)
    )

# --- REPLACE THE ATTENTION CLASS ---
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        # Calculate how many times to repeat keys/values
        self.n_rep = self.n_heads // self.n_kv_heads 
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape
        
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Transpose to [B, H, S, D]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # --- FIX: REPEAT KV HEADS ---
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)
        
        # Now shapes match: [B, 4, S, D] vs [B, 4, S, D]
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output)

# --- 3. The Transformer ---
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CausalSelfAttention(args),
                'feed_forward': MLP(args),
                'attention_norm': RMSNorm(args.dim),
                'ffn_norm': RMSNorm(args.dim),
            }) for _ in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Initialize RoPE cache
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta
        )

    def forward(self, input_ids):
        B, S = input_ids.shape
        
        # Handle RoPE device movement
        if self.freqs_cis.device != input_ids.device:
            self.freqs_cis = self.freqs_cis.to(input_ids.device)
            
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[:S]
        
        for layer in self.layers:
            # 1. Attention Block
            h_norm = layer['attention_norm'](h)
            h = h + layer['attention'](h_norm, freqs_cis)
            
            # 2. FFN Block (SwiGLU)
            h_norm = layer['ffn_norm'](h)
            h = h + layer['feed_forward'](h_norm)
            
        return self.output(self.norm(h))