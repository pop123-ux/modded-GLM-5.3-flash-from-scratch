"""
Educational GLM-Style Hybrid Transformer - inspired by Vuk Rosic's glm-5.3-flash-from-scratch repository

This is not a 1:1 reproduction of GLM-5.3-Flash.
It is a from-scratch educational implementation inspired by
modern language-model architectures.

The model combines several concepts, including Mixture-of-Experts
(MoE), hybrid attention, multiple residual streams, and learned
stream routing, with an emphasis on making the underlying PyTorch
operations and tensor transformations explicit and inspectable.

See DOCUMENTATION.md for a detailed explanation of the architecture,
implementation decisions, experiments, and educational objectives.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any


def trace_tensor(
    name: str,
    x: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """
    Educational tensor-shape tracer.

    Prints:
        tensor name
        shape
        dtype
        device
        number of elements
        memory footprint
    """

    if not enabled:
        return x

    numel = x.numel()

    # Approximate memory footprint.
    memory_kb = numel * x.element_size() / 1024

    print(
        f"[TRACE] {name:<30} "
        f"shape={str(tuple(x.shape)):<22} "
        f"dtype={str(x.dtype):<12} "
        f"device={str(x.device):<8} "
        f"elements={numel:<10} "
        f"memory={memory_kb:.2f} KB"
    )

    return x

@dataclass(frozen=True) # The python dataclass is a built-in tool that automaticall generates boilerplate methods for data-storing classes. The frozen=True argument makes the dataclass immutable, preventing fields from being modified after creation and making the object safely hashable
class ModelConfig:
    """Standard Transformer Architecture Components"""
    vocab_size: int = 256 + 4 * 1 # Using byte-level tokenization we need 256 starter tokens + <pad> + <bos> + <eos> + <sep>
    dim: int = 128 # The size or length of a vector used to represent data | With the default configuration, every token is represented by 128 numbers | Scaling of the dim variable allows the model to capture more complex relationships between words | Observation: Every time we double the dim, the amount of memory and compute required by the model's linear layers increases close to quadratically (x**2) | Essentially the width of the model
    layers: int = 12 # The total number of model layers | Transform raw input tokens into complex concepts | Essentially the depth of the model
    heads: int = 4 # Sub-units within a single layer that split the model's total dimension to look at a sentence from multiple perspectives at the same time -> Multi-Head Attention | Dimension per head -> head_dim: int = dim / heads = 32 (size of each individual head) A vector of 128 numbers is sliced into 4 vectors of 32 numbers.
    """Mixture-Of-Experts Components"""
    expert_hidden: int = 4 * 128 # 4 x dim (hidden_size) = 512 | Expert number of neurons - part of MoE architecture
    experts: int = 4 # Number of experts used - also part of the MoE architecture
    top_k: int = 2 # Top 2 Mixture of Experts (MoE) routing - the exact number of expert networks a single token will be sent to at every MoE layer.
    """Residual-Streams Component"""
    streams: int = 8 # Learned stream mixing inspired by hyper-connections; this is not the manifold-constrained algorithm | Enables the raw internal matrices to transform the standard layout into a much larger multidimensional tensor [batch_size, sequence_length, number_of_streams, dim]
    """Block Sparse-Attention Components"""
    sparse_window: int = 16 # The number of continuous, back-to-back tokens the model can look at during a single hop | Enables the model to capture local context
    sparse_stride: int = 16 # Dictates the step size across the sequence history | If sparse_stride: 16, the model skips over 16 tokens before activating its next attention window | Allows the model to look far back into its past memory without causing the GPU to crash
    """Inference"""
    max_sequence_length: int = 512 # Absolute maximum number of tokens a model can process or generate in a single prompt session
    
    # ------------------------------------------------------------------
    # Educational / Diagnostics
    # ------------------------------------------------------------------
    trace_shapes: bool = False
    diagnostics: bool = False
    residual_alpha: float = 0.01
    residual_dropout: float = 0.1
    attention_mode: str = "hybrid"  # hybrid or full causal SDPA
    sparse_backend: str = "gather"  # gather or masked SDPA
    sparse_every: int = 4
    residual_mode: str = "streams"  # streams or standard
    qk_norm: bool = False
    embedding_skip: bool = False
    logit_softcap: float = 0.0
    padding_aware_routing: bool = False
    linear_chunk_size: int = 0
    linear_checkpoint_chunks: bool = False
    ffn_mode: str = "moe"  # moe or dense (active expert-width control)
    dense_hidden: int = 0  # 0 => (top_k + shared expert) * expert_hidden
    router_bias_rate: float = 0.0
    router_bias_limit: float = 2.0
    future_heads: int = 0  # Additional offsets: 2, 3, ... bytes ahead

    def __post_init__(self):
        positive = ("vocab_size", "dim", "layers", "heads", "expert_hidden", "experts",
                    "top_k", "streams", "sparse_window", "sparse_stride", "max_sequence_length", "sparse_every")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("model dimensions and routing counts must be positive")
        if self.dim % self.heads or self.head_dim % 2:
            raise ValueError("dim must divide into heads with even RoPE head_dim")
        if self.top_k > self.experts:
            raise ValueError("top_k must not exceed experts")
        if self.attention_mode not in ("hybrid", "sdpa") or self.sparse_backend not in ("gather", "sdpa"):
            raise ValueError("invalid attention backend")
        if self.ffn_mode not in ("moe", "dense"):
            raise ValueError("invalid ffn_mode")
        if self.linear_chunk_size < 0 or self.dense_hidden < 0 or self.future_heads < 0:
            raise ValueError("chunk size, dense width, and future head count must be nonnegative")
        if self.linear_checkpoint_chunks and not self.linear_chunk_size:
            raise ValueError("checkpointed linear attention requires a positive chunk size")
        if self.router_bias_rate < 0 or self.router_bias_limit <= 0:
            raise ValueError("invalid router bias settings")
        if self.ffn_mode == "dense" and self.router_bias_rate:
            raise ValueError("dense FFN has no router to balance")
        if self.future_heads >= self.max_sequence_length:
            raise ValueError("future head count must be smaller than context")
        if self.residual_mode not in ("streams", "standard"):
            raise ValueError("invalid residual mode")
        if not 0 <= self.residual_dropout < 1 or self.logit_softcap < 0:
            raise ValueError("invalid dropout or logit softcap")

    @property
    def head_dim(self):
        return self.dim // self.heads

    @property
    def topk(self):
        return self.top_k

    @property
    def num_streams(self):
        return self.streams
    
    def to_dict(self) -> dict[str, int]:
        return asdict(self)

"""https://arxiv.org/abs/2104.09864"""
def apply_rot(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to [B, T, H, H_dim] | Multi-Head Attention Standard"""
    length, width = q.shape[1], q.shape[-1] # Index 1 -> Sequence Length (length) | Index -1 -> Head Dimension (width)
    if width % 2:
        raise ValueError("RoPE head dimension must be divisible by 2")
    positions = torch.arange(length, device=q.device, dtype=torch.float32) # Shape [seq_len,]
    frequencies = 1.0 / (10000 ** (torch.arange(0, width, 2, device=q.device).float() / width)) # Shape [width // 2,] | Sinusoidal Positional Encoding
    angles = positions[:, None] * frequencies[None, :] # Shape [seq_len, width // 2] | Make both tensors fit for broadcasting by adding another dimension
    cos = angles.cos()[None, :, None, :].to(q.dtype) # [None, :, None, :] reshapes the 2D matrix into a 4D tensor to match the exact structural layout of the Query and Key in Multi-Head Attention
    sin = angles.sin()[None, :, None, :].to(q.dtype) # Basically sin([1, seq_len, 1, width // 2])
    
    """Takes either Q or K tensor and rotates pairs of adjacent numbers in its hidden dimension using the cos and sin tensors calculated"""
    def rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2] # Splits the width into two halves based on index positions
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2) # Shape even acts as X coordinates, and odd acts as Y coordinates | The flatten operation combines the last two dimensions ( (width // 2) * 2 = width ) Shape transforms [batch_size, seq_len, num_heads, width // 2, 2] -> [batch_size, seq_len, num_heads, width]
        
    return rotate(q), rotate(k)

"""https://arxiv.org/abs/1910.07467"""
class RotaryEmbedding(nn.Module):
    """One bounded, nonpersistent cache per attention layer; rebuilt on device/dtype changes."""
    def __init__(self):
        super().__init__()
        self._cache = None
    def forward(self, q, k):
        t, d = q.shape[1], q.shape[-1]
        key = (t, d, q.device, q.dtype)
        if self._cache is None or self._cache[0] != key:
            positions = torch.arange(t, device=q.device, dtype=torch.float32)
            frequencies = 10000 ** (-torch.arange(0,d,2,device=q.device,dtype=torch.float32)/d)
            angles = positions[:,None] * frequencies[None,:]
            self._cache = (key, angles.cos()[None,:,None,:].to(q.dtype), angles.sin()[None,:,None,:].to(q.dtype))
        _, cos, sin = self._cache
        def rotate(x):
            even, odd = x[...,0::2], x[...,1::2]
            return torch.stack((even*cos-odd*sin, even*sin+odd*cos),-1).flatten(-2)
        return rotate(q), rotate(k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # Essentially the gamma from the outputs normalization using RMSNorm formula
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight

class LinearAttention(nn.Module):
    """Causal positive-feature linear attention with recurrent prefix state."""
    def __init__(self, config: ModelConfig, trace_shapes: bool = False,):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.chunk_size = config.linear_chunk_size
        self.checkpoint_chunks = config.linear_checkpoint_chunks
        self.trace_shapes = trace_shapes or config.trace_shapes
        self.rope = RotaryEmbedding()
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.qkv = nn.Linear(in_features=config.dim, out_features=config.dim * 3, bias=False) # [batch_size, seq_len, config.dim] -> [batch_size, seq_len, config.dim * 3]
        self.out = nn.Linear(in_features=config.dim, out_features=config.dim, bias=False) # Final projection layer of the attention block, often called the output projection/dense layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        trace_tensor("LinearAttention input", x, self.trace_shapes)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        trace_tensor("Q projection", q, self.trace_shapes)
        trace_tensor("K projection", k, self.trace_shapes)
        trace_tensor("V projection", v, self.trace_shapes)
                     
        """Reshaping Q, K, V tensors to respect the Multi-Head Attention Layout."""
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch, length, self.heads, self.head_dim)
        
        trace_tensor("Q reshaped", q, self.trace_shapes)
        trace_tensor("K reshaped", k, self.trace_shapes)
        trace_tensor("V reshaped", v, self.trace_shapes)
        
        q, k = self.rope(self.q_norm(q), self.k_norm(k))
        
        trace_tensor("Q after RoPE", q, self.trace_shapes)
        trace_tensor("K after RoPE", k, self.trace_shapes)
        
        q = F.elu(q) + 1.0 # The Exponential Linear Unit (ELU) activation function outputs values down to -1.0. By adding 1.0, we force every single number in q and k to be >= 0
        k = F.elu(k) + 1.0
        if self.chunk_size:
            from .linear import chunked_linear_attention
            output = chunked_linear_attention(q, k, v, self.chunk_size, self.checkpoint_chunks)
            return self.out(output.reshape(batch, length, dim))
        kv = torch.einsum("bthd,bthe->bthde", k, v).cumsum(dim=1) # Shape [batch, length, heads, head_dim_k, head_dim_v] Computes the KV-cache and enforces causality (tokens can only look at past tokens, not future ones)
        
        trace_tensor(
                    "Cumulative KV state",
                    kv,
                    self.trace_shapes,
        )
        
        k_prefix = k.cumsum(dim=1) # Shape [batch, length, heads, head_dim_k] Computes the cumulative sum of the Key tensor alone along the Time axis | Storing the running sum of all past keys allows us to track the total "weight" of the history up to the current token
        n = torch.einsum('bthd,bthde->bthe', q, kv) # Shape [batch, length, heads, head_dim_v] (this line basically computes: [head_dim_k] * [head_dim_k, head_dim_v] = [head_dim_v]) Multiplies the Q (Query) by the running historical memory matrix (kv)
        d = torch.einsum('bthd,bthd->bth', q, k_prefix).unsqueeze(-1) # Shape [batch, length, heads, 1] | Computes dot product between the current Q (Query) and the accumulated historical Keys (k_prefix) at each time step | Acts as the replacement for the Softmax denominator, scaling the features so that tokens don't explode in value as the text gets longer
        """Complete the causal attention lookup in a perfectly linear time-frame"""
        output = n / d.clamp_min(1e-6) # Shape [batch, length, self.heads, self.head_dim] | Prevents division by 0 error
        
        trace_tensor(
            "Linear attention output",
            output,
            self.trace_shapes,
        )
        
        output = self.out(
            output.reshape(batch, length, dim)
        )

        trace_tensor(
            "LinearAttention projected output",
            output,
            self.trace_shapes,
        )
        
        return output

class SparseAttention(nn.Module):
    """Gathered causal attention over a local window plus strided anchors."""
    
    def __init__(self, config: ModelConfig, trace_shapes: bool = False,):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.trace_shapes = trace_shapes or config.trace_shapes
        self.backend = config.sparse_backend
        self._index_cache = None
        self._mask_cache = None
        self.rope = RotaryEmbedding()
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.window = config.sparse_window
        self.stride = config.sparse_stride
        self.qkv = nn.Linear(in_features=config.dim, out_features=config.dim * 3, bias=False)
        self.out = nn.Linear(in_features=config.dim, out_features=config.dim, bias=False)
    
    """Method that inputs the total sequence length (length) and the destination hardware, returning a tuple of 2 tensors: one for row indices and one for column indices"""
    def _indices(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (length, device)
        if self._index_cache is not None and self._index_cache[0] == key:
            return self._index_cache[1]
        rows: list[list[int]] = [] # Which tokens can interact
        for position in range(length):
            anchors = list(range(0, position + 1, self.stride)) # Autoregressive encoding: a token can only look at the past and itself, never into the future | +1 ensures the current token is included in the range
            local = list(range(max(0, position + 1 - self.window), position + 1)) # Generates a list of indices representing the immediate past neighbors of the current position | max(0, ...) prevents the lookback window from sliding into negative numbers if the current position is near the start of the text
            rows.append(sorted(set(anchors + local))) # Merges the strided tokens (anchors) and the immediate neighbors (local) for the current row
        width = max(map(len, rows)) # Finds the absolute maximum number of tokens any single row is attending to | Longest sub-list in the entire batch
        indices = torch.zeros((length, width), dtype=torch.long, device=device)
        valid = torch.zeros((length, width), dtype=torch.bool, device=device)
        for row, values in enumerate(rows):
            indices[row, : len(values)] = torch.tensor(values, device=device)
            valid[row, : len(values)] = True # Marks the real data slots as True and keeps the padding slots as False
        self._index_cache = (key, (indices, valid))
        return indices, valid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        
        trace_tensor(
            "SparseAttention input",
            x,
            self.trace_shapes,
        )
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Trace the already-computed projections; avoid repeating QKV matmul.
        trace_tensor("Sparse Q", q, self.trace_shapes)
        
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        q, k = self.rope(self.q_norm(q), self.k_norm(k))
        q = q.transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        k = k.transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        
        trace_tensor("Sparse Q", q, self.trace_shapes)
        trace_tensor("Sparse K", k, self.trace_shapes)
        trace_tensor("Sparse V", v, self.trace_shapes)
        
        indices, valid = self._indices(length, x.device) # self._indices() produces Shape [length, k]
        
        if self.trace_shapes:
            print(
                f"[TRACE] Sparse indices shape       "
                f"{tuple(indices.shape)}"
            )

            print(
                f"[TRACE] Average attention positions "
                f"{valid.float().sum(dim=-1).mean().item():.2f}"
            )
            
        if self.backend == "sdpa":
            key = (length, x.device)
            if self._mask_cache is None or self._mask_cache[0] != key:
                mask = torch.zeros(length, length, dtype=torch.bool, device=x.device)
                # Construct only valid pairs; padded indices must not overwrite anchors.
                rows = torch.arange(length, device=x.device)[:, None].expand_as(indices)
                mask[rows[valid], indices[valid]] = True
                self._mask_cache = (key, mask)
            outputs = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask_cache[1])
            return self.out(outputs.transpose(1, 2).reshape(batch, length, dim))
        gathered_k = k[:, :, indices, :] # Shape [batch, self.heads, length, k, self.head_dim]
        gathered_v = v[:, :, indices, :] # Shape [batch, self.heads, length, k, self.head_dim]
        
        trace_tensor(
            "Gathered K",
            gathered_k,
            self.trace_shapes,
        )

        trace_tensor(
            "Gathered V",
            gathered_v,
            self.trace_shapes,
        )
        
        scores = torch.einsum('bhtd,bhtkd->bhtk', q, gathered_k) * self.head_dim**-0.5 # Shape [batch, self.heads, length, k] | torch.einsum(...) does the dot product between q and gathered_k | * self.head_dim **-0.5 does the scaling -> Scaled Dot-Product in order to prevent the gradients from exploding
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min) # Masks the attention scores (logits) by setting padded or invalid positions to -infinity, ensuring they receive exactly 0% attention weight after the Softmax operation
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype) # ransforms the attention logits into probability weights (0.0 to 1.0)
        outputs = torch.einsum('bhtk,bhtkd->bhtd', weights, gathered_v) # Shape [batch, self.heads, length, self.head_dim]
        output = self.out(outputs.transpose(1, 2).reshape(batch, length, dim)) # Shape [batch, length, dim = heads * head_dim]
    
        trace_tensor(
            "SparseAttention output",
            output,
            self.trace_shapes,
        )
        
        return output
class CausalSelfAttention(nn.Module):
    """Full causal SDPA ablation; kernel choice remains PyTorch/device dependent."""
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.head_dim
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RotaryEmbedding()
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = [a.view(b, t, self.heads, self.head_dim) for a in self.qkv(x).chunk(3, -1)]
        q, k = self.rope(self.q_norm(q), self.k_norm(k))
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Expert(nn.Module):
    """Expert class that SparseMixtureOfExperts will use"""
    def __init__(self, dim: int, hidden: int, trace_shapes: bool = False,):
        super().__init__()
        self.trace_shapes = trace_shapes
        self.up = nn.Linear(in_features=dim, out_features=hidden * 2, bias=False)
        self.down = nn.Linear(in_features=hidden, out_features=dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        trace_tensor(
            "Expert input",
            x,
            self.trace_shapes,
        )
        
        up = self.up(x)
        
        trace_tensor(
            "Expert up projection",
            up,
            self.trace_shapes,
        )
        
        gate, value = up.chunk(2, dim=-1)

        gated = F.silu(gate) * value

        trace_tensor(
            "Expert gated activation",
            gated,
            self.trace_shapes,
        )

        output = self.down(gated)

        trace_tensor(
            "Expert output",
            output,
            self.trace_shapes,
        )

        return output
    
r""" Visual representation of the Expert class

              [ In ] (dim)
                 |
             [ self.up ]
                 |
           [ hidden * 2 ]
          /         \   (.chunk)
    [ gate ]      [ value ] (hidden)
       |              |
       |              |
   [ F.silu ]         |
       \              /
        \--->(*)<---/ (Gated Linear Unit)
              |
          [ hidden ]
              |
        [ self.down ]
              |
           [ Out ] (dim)
"""


class SparseMixtureOfExperts(nn.Module):
    """Top-2 routed experts plus one always-active shared expert - Trending approach"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Linear(in_features=config.dim, out_features=config.experts, bias=False) # Defines the router that will route the tokens to the chosen experts
        self.experts = nn.ModuleList(
            [Expert(config.dim, config.expert_hidden) for _ in range(config.experts)] # nn.ModuleList is a torch.nn submodule that holds submodules in a list format
        )
        self.shared = Expert(config.dim, config.expert_hidden) # The shared expert
        self.top_k = config.top_k
        self.diagnostics = config.diagnostics
        self.last_router_probs = None
        self.last_usage = None
        self.aux_loss = None
        self.z_loss = None
        self.bias_rate = config.router_bias_rate
        self.bias_limit = config.router_bias_limit
        self.pending_usage = None
        self._last_valid_tokens = 0
        if self.bias_rate:
            self.register_buffer("router_bias", torch.zeros(config.experts))

    def forward(self, x: torch.Tensor, return_usage: Optional[bool] = True,
                token_mask: Optional[torch.Tensor] = None):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        if token_mask is not None:
            if token_mask.shape != x.shape[:-1] or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be boolean with shape [batch, time]")
            positions = token_mask.reshape(-1).nonzero(as_tuple=True)[0]
            active = flat[positions]
        else:
            positions = None
            active = flat
        self._last_valid_tokens = active.shape[0]
        usage = flat.new_zeros(len(self.experts), dtype=torch.float32)
        if not active.shape[0]:
            self.aux_loss = self.router.weight.sum() * 0
            self.z_loss = self.aux_loss
            self.pending_usage = usage
            output = x * 0
            if self.diagnostics:
                self.last_router_probs = flat.new_zeros(flat.shape[0], len(self.experts)).cpu()
                self.last_usage = usage.cpu()
            return (output, usage) if return_usage else output
        logits = self.router(active)
        probabilities = logits.float().softmax(dim=-1)
        selection_scores = logits.float() + self.router_bias if self.bias_rate else logits
        top_indices = selection_scores.topk(self.top_k, dim=-1).indices
        # Bias changes selection only: mixture weights use the original logits.
        top_weights = logits.gather(-1, top_indices).float().softmax(dim=-1).to(active.dtype)
        result = self.shared(active)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(top_indices == expert_id)
            if not rows.numel():
                continue
            contribution = expert(active[rows]) * top_weights[rows, slots, None]
            result.index_add_(0, rows, contribution.to(result.dtype))
            usage[expert_id] = rows.numel()
        usage = usage / (active.shape[0] * self.top_k)
        self.pending_usage = usage.detach()
        self.aux_loss = len(self.experts) * (usage.detach() * probabilities.mean(0)).sum()
        self.z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
        if positions is not None:
            output = flat.new_zeros(flat.shape).index_copy(0, positions, result)
        else:
            output = result
        if self.diagnostics:
            probs = probabilities.detach()
            if positions is not None:
                probs = probs.new_zeros(flat.shape[0], len(self.experts)).index_copy(0, positions, probs)
            self.last_router_probs = probs.cpu()
            self.last_usage = usage.detach().cpu()
        output = output.view(original_shape)
        return (output, usage) if return_usage else output

    @torch.no_grad()
    def update_bias(self):
        """Explicit post-update controller; validation/inference never mutates bias."""
        if self.bias_rate and self.training and self._last_valid_tokens and self.pending_usage is not None:
            target = 1.0 / len(self.experts)
            delta = (target - self.pending_usage).sign()
            delta -= delta.mean()
            self.router_bias.add_(delta, alpha=self.bias_rate)
            self.router_bias.clamp_(-self.bias_limit, self.bias_limit)
            self.pending_usage = None


class DenseFeedForward(nn.Module):
    """Dense SwiGLU control with roughly matched active expert matrix work."""
    def __init__(self, config):
        super().__init__()
        hidden = config.dense_hidden or (config.top_k + 1) * config.expert_hidden
        self.dense = Expert(config.dim, hidden, config.trace_shapes)
        self.experts = nn.ModuleList()  # Parameter-count/diagnostic compatibility.
        self.last_router_probs = self.last_usage = None
        self.aux_loss = self.z_loss = None
        self.expert_count = config.experts
    def forward(self, x, return_usage=True, token_mask=None):
        output = self.dense(x)
        if token_mask is not None:
            output = output * token_mask[..., None]
        self.aux_loss = output.sum() * 0
        self.z_loss = self.aux_loss
        usage = output.new_zeros(self.expert_count, dtype=torch.float32)
        return (output, usage) if return_usage else output
    def update_bias(self):
        pass


class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = (CausalSelfAttention(config) if config.attention_mode == "sdpa" else
                          SparseAttention(config) if sparse else LinearAttention(config))
        self.moe = SparseMixtureOfExperts(config) if config.ffn_mode == "moe" else DenseFeedForward(config)
    
    """Pre-layer normalization and standard residual (skip) connections across two primary processing phases"""
    def forward(self, x: torch.Tensor, token_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x)) # Phase 1: Attention Processing | x + self.attention(...) represents a Residual Connectioon that prevents vanishing gradients during backpropagation
        ffn, usage = self.moe(self.ffn_norm(x), token_mask=token_mask) # Phase 2: Feed-Forward Processing (MoE) | The updated tensor x is normalized again using self.ffn_norm, which is then passed to the MoE layer, which returns ffn: the actual processed features output by the selected active experts; usage: a structural tensor tracking how frequently individual experts were chosen
        return x + ffn, usage
    
class HyperConnection(nn.Module):
    """Eight residual streams w/ learned input mixing and output routing."""
    
    def __init__(self, config: ModelConfig, block: HybridBlock, trace_shapes: bool = False):
        super().__init__()
        self.block = block
        self.trace_shapes = trace_shapes or config.trace_shapes
        self.diagnostics = config.diagnostics
        self.residual_mode = config.residual_mode
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams)) # Tensor representing the starter input logits evenly spaced from 0.1 to -0.1 with a size of 8 | nn.Parameter means that PyTorch will adjust the values during backpropagation
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams)) # Tensor representing the starter output logits evenly spaced from 0.1 to -0.1 with a size of 8 | nn.Parameter means that PyTorch will adjust the values during backpropagation
    
        self.alpha = nn.Parameter(torch.tensor(config.residual_alpha)) # Scalar initialized initially with 0.01, it reduces the influence of the block so that the network will perform similar to a clean residual connection
        self.dropout = nn.Dropout(config.residual_dropout) # Randomly zeroes out 10% of the generated features during training, forcing the 8 residual streams to learn independent representations
    
        # --------------------------------------------------------------
        # Residual-stream telemetry
        # --------------------------------------------------------------

        self.last_streams: Optional[torch.Tensor] = None
        self.last_input_weights: Optional[torch.Tensor] = None
        self.last_output_weights: Optional[torch.Tensor] = None
        self.last_stream_norms: Optional[torch.Tensor] = None
        self.last_stream_similarity: Optional[torch.Tensor] = None
        
    """Residual streams forward pass, expects an input tensor streams of shape [Batch, Tokens, Streams, Dim]"""
    def forward(self, streams: torch.Tensor, token_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        
        trace_tensor(
        "HyperConnection input",
        streams,
        self.trace_shapes,
        )
        
        # --------------------------------------------------------------
        # Input stream mixing
        # --------------------------------------------------------------

        if self.residual_mode == "standard":
            transformed, usage = self.block(streams[:, :, 0], token_mask=token_mask)
            return transformed.unsqueeze(2).expand_as(streams), usage
        input_weights = torch.softmax( # Converts the learnable input logits into a probability distribution across the residual streams | Shape: [S] where S = number of residual streams | Because softmax is applied across dim=0, all input stream weights sum to 1.0
        self.input_logits,
        dim=0,
        )

        if self.diagnostics:
            self.last_input_weights = (
            input_weights.detach().cpu() # Detaches the routing weights from the computation graph and moves them to the CPU so they can be inspected later without affecting gradient calculations
        )

        mixed = torch.einsum(
            "s,btsd->btd",
            input_weights,
            streams, # torch.einsum performs a weighted reduction across the stream dimension | Each stream [B, T, D] is multiplied by its corresponding input weight and then summed together | Shape transformation: [B, T, S, D] -> [B, T, D] | This compresses the S parallel residual streams into one representation that can be processed by the main transformer block
        )

        trace_tensor(
            "Mixed residual streams",
            mixed,
            self.trace_shapes,
        )

        # --------------------------------------------------------------
        # Main transformer block
        # --------------------------------------------------------------

        transformed, usage = self.block(
            mixed, token_mask=token_mask
        ) # Passes the mixed representation tensor directly into HybridBlock, processing the combined representation through Attention and MoE | transformed has shape [B, T, D] | usage contains the MoE expert utilization statistics

        trace_tensor(
            "Transformed block output",
            transformed,
            self.trace_shapes,
        )

        # --------------------------------------------------------------
        # Output routing
        # --------------------------------------------------------------

        output_weights = torch.softmax( # Converts the learnable output logits into a probability distribution across the residual streams | Shape: [S] | The weights determine how strongly the transformed representation contributes to each individual residual stream
            self.output_logits,
            dim=0,
        )

        if self.diagnostics:
            self.last_output_weights = (
            output_weights.detach().cpu() # Stores a detached CPU copy of the output routing probabilities for telemetry, visualization, and educational inspection
        )

        features = (
            transformed.unsqueeze(2) # Changes [B, T, D] -> [B, T, 1, D] by inserting a new dimension at index 2 so the transformed representation can be distributed across the S residual streams
            * output_weights[
                None,
                None,
                :,
                None,
            ] # Reshapes [S] -> [1, 1, S, 1] | Broadcasting expands the tensors to [B, T, S, D], meaning every stream receives the same transformed representation but scaled by its own learned output-routing weight
        )

        trace_tensor(
            "Routed stream features",
            features,
            self.trace_shapes,
        )

        # --------------------------------------------------------------
        # Residual update
        # --------------------------------------------------------------

        streams = (
            streams
            + self.alpha * self.dropout(features) # Applies dropout to the newly generated stream features, scales them using the learnable alpha barrier, and adds them to the original streams through a residual connection | alpha controls the strength of the new information injected into the existing streams | Shape remains [B, T, S, D]
        )

        trace_tensor(
            "Updated residual streams",
            streams,
            self.trace_shapes,
        )

        # --------------------------------------------------------------
        # Telemetry
        # --------------------------------------------------------------

        if self.diagnostics:
            self.last_alpha = (
            self.alpha.detach().cpu() # Stores a detached copy of the learnable residual scaling factor for later inspection and visualization
        )

        if self.diagnostics:
            self.last_usage = (
            usage.detach().cpu() # Stores a detached CPU copy of the MoE expert utilization statistics for telemetry without keeping the computation graph alive
        )

        # --------------------------------------------------------------
        # Return
        # --------------------------------------------------------------

        if self.diagnostics:
            snapshot = streams.detach().float()
            self.last_stream_norms = snapshot.square().mean(-1).sqrt().cpu()
            normalized = F.normalize(snapshot, dim=-1)
            self.last_stream_similarity = torch.einsum("btsd,btud->su", normalized, normalized).div(snapshot.shape[0] * snapshot.shape[1]).cpu()
        return streams, usage # Returns the updated multi-stream representation [B, T, S, D] together with the MoE expert usage statistics [experts]
        
class ModdedGLM53FlashFromScratch(nn.Module):
    def __init__(self, config: ModelConfig, trace_shapes: Optional[bool] = None,):
        super().__init__()
        if config.dim % config.heads: # This safety check guarantees that the hidden dimension (config.dim) can be perfectly split across the multi-head attention mechanisms (config.heads) | If it is not divisible, Multi-Head Attention cannot evenly partition vector spaces, which throws runtime errors
            raise ValueError("dim must be divisible by heads")
        self.config = config
        self.trace_shapes = (
            config.trace_shapes
            if trace_shapes is None
            else trace_shapes
        )
        self.embedding = nn.Embedding(config.vocab_size, config.dim) # Maps raw integer token IDs into continuous dense vector representations of size config.dim, preparing them for neural processing
        self.layers = nn.ModuleList([
            HyperConnection(config, HybridBlock(config, sparse=((index + 1) % config.sparse_every == 0))) # Instantiates the model's structural depth (config.layers) | by using ((index + 1) % 4 == 0), every 4th layer is initializes with a SparseAttention mechanisms inside of its HybridBlock | The remaining 3 intermediate layers use LinearAttention, mimicking structural techniques found in state-of-the-art models like GLM-5.3-Flash to balance long-context computational costs | Each HybridBlock is wrapped inside a HyperConnection instance to enforce multi-stream routing and prevent representation decay across the layers
            for index in range(config.layers)   # Let i be the current layer number -> if i % 4, Layer i = SparseAttention else Layer i = LinearAttention
        ])
        self.embedding_skip_scale = nn.Parameter(torch.zeros(config.layers)) if config.embedding_skip else None
        self.final_norm = RMSNorm(config.dim) # Before generating vocabulary predictions, the hidden states pass through a final Root Mean Square Normalization layer to standardize data magnitudes across the batch
        self.output = nn.Linear(in_features=config.dim, out_features=config.vocab_size, bias=False) # This linear projection shifts the output dimension from the model's internal hidden space back to a massive array matching the vocabulary length, generating raw logit scores for every possible token | dim -> vocab_size
        self.future_predictors = nn.ModuleList([nn.Linear(config.dim, config.dim, bias=False) for _ in range(config.future_heads)])
        self._future_hidden = None
        self.output.weight = self.embedding.weight # This memory optimization trick shares the parameter weights between the input embedding layer and the final output prediction layer, cutting down overall model parameter count significantly, enforcing semantic alignment (a token's representation when reading text matches its representation when generating text)
        self.apply(self._init_weights) # Recursively sweep through every submodule defined in this class and apply a custom weight initialization routine to ensure parameters start with optimal distributions before training
        
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # Fills module.weight with values drawn from the normal distribution, with a mean of 0 and custom std of 0.02
            
    def forward_embeddings(self, embedded: torch.Tensor, token_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the language model from precomputed token-like embeddings."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded inputs must have shape [batch, time, dim] (B, T, D)")
        if embedded.shape[1] < 1:
            raise ValueError("embedded inputs must contain at least one token")
        if embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("embedded input exceeds max_sequence_length")
        if token_mask is not None and (token_mask.shape != embedded.shape[:2] or token_mask.dtype != torch.bool):
            raise ValueError("token_mask must be boolean with shape [batch, time]")
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.streams, -1) # Multi-Stream Setup; New shape [B, T, 8, D] | embedded.unsqueeze(2) transforms [B, T, D] -> [B, T, 1, D] | .expand(-1, -1, self.config.streams, -1) copies the same entry embeddings from each 8 parallel streams | It allocates memory
        usages = [] # Empty list to collect the usage logs generated by each SparseMixtureOfExperts routing layer
        for index, layer in enumerate(self.layers): # Loops through the stacked layer infrastructure | Each layer (HyperConnections instance) takes the [B, T, 8, D] streams tensor, passes a compressed version through its inner HybridBlock, updates the streams, and returns the newly modified streams along with a usage tensor tracking expert activation
            if self.embedding_skip_scale is not None:
                streams = streams + self.embedding_skip_scale[index] * embedded.unsqueeze(2)
            streams, usage = layer(streams, token_mask=token_mask)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2)) # Compresses the tensor down to [Batch, Time, Dim], collapsing the multi-stream dimension by averaging the 8 lanes, combining all independently learned structural & semantic properties into a single unified context vector
        self._future_hidden = hidden if self.config.future_heads else None
        logits = self.output(hidden)
        if self.config.logit_softcap:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        return logits, torch.stack(usages) # self.output projects the hidden dimension (D) up to the full vocab size (Vocab), yielding a final prediction tensor of shape [Batch, Time, Vocab] containing raw token generation logits | torch.stack(usages) converts the list of separate layer metrics into a single, cohesive PyTorch tensor, for detached utilization diagnostics; differentiable router losses are available separately via router_losses()
    
    def plot_stream_diagnostics(
    self,
    layer_idx: int = 0,
    ):
        """
        Visualize:
            1. Input mixing weights
            2. Output routing weights
            3. Average stream norms
            4. Stream cosine similarity
        """

        import matplotlib.pyplot as plt

        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError("Invalid layer index")

        layer = self.layers[layer_idx]

        input_weights = layer.last_input_weights
        output_weights = layer.last_output_weights
        norms = layer.last_stream_norms
        similarity = layer.last_stream_similarity

        if input_weights is None:
            raise RuntimeError(
                "Run a forward pass first."
            )

        # --------------------------------------------------------------
        # Average norms across batch/time
        # --------------------------------------------------------------

        mean_norms = norms.mean(dim=(0, 1))

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 9),
        )

        # --------------------------------------------------------------
        # Input mixing
        # --------------------------------------------------------------

        axes[0, 0].bar(
            range(len(input_weights)),
            input_weights.numpy(),
        )

        axes[0, 0].set_title(
            "Input Stream Mixing Weights"
        )

        axes[0, 0].set_xlabel("Stream")
        axes[0, 0].set_ylabel("Weight")

        # --------------------------------------------------------------
        # Output routing
        # --------------------------------------------------------------

        axes[0, 1].bar(
            range(len(output_weights)),
            output_weights.numpy(),
        )

        axes[0, 1].set_title(
            "Output Stream Routing Weights"
        )

        axes[0, 1].set_xlabel("Stream")
        axes[0, 1].set_ylabel("Weight")

        # --------------------------------------------------------------
        # Stream norms
        # --------------------------------------------------------------

        axes[1, 0].bar(
            range(len(mean_norms)),
            mean_norms.numpy(),
        )

        axes[1, 0].set_title(
            "Average Residual Stream Norm"
        )

        axes[1, 0].set_xlabel("Stream")
        axes[1, 0].set_ylabel("RMS")

        # --------------------------------------------------------------
        # Similarity matrix
        # --------------------------------------------------------------

        image = axes[1, 1].imshow(
            similarity.numpy(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )

        axes[1, 1].set_title(
            "Residual Stream Cosine Similarity"
        )

        axes[1, 1].set_xlabel("Stream")
        axes[1, 1].set_ylabel("Stream")

        fig.colorbar(
            image,
            ax=axes[1, 1],
        )

        fig.suptitle(
            f"HyperConnection Diagnostics — Layer {layer_idx}",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()

    def plot_usages(self, input_ids: torch.Tensor):
        import matplotlib.pyplot as plt
        import numpy as np
        
        self.eval()
        with torch.no_grad():
            embedded = self.embedding(input_ids) * (self.config.dim ** 0.5)
            _, usages = self.forward_embeddings(embedded=embedded)
        if usages.numel() == 0:
            print('No MoE usage telemetry collected.')
            return
        usages_np = usages.detach().cpu().numpy() # Move the tensor to CPU and convert to NumPy array
        num_layers = usages_np.shape[0]
        num_experts = usages_np.shape[1] if usages_np.ndim > 1 else 1
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if usages_np.ndim == 2:
            x_axis = np.arange(num_layers)
            for expert_idx in range(num_experts):
                ax.plot(
                    x_axis, 
                    usages_np[:, expert_idx], 
                    label=f'Expert {expert_idx}', 
                    marker='o', 
                    linewidth=1.5
                )
            ax.set_xlabel('Layer Index (Model Depth)')
            ax.set_xticks(x_axis)
        else:
            # Fallback if the usage tracking is a flat sequence over time/iterations
            ax.plot(usages_np, label='Global MoE Usage', color='blue', linewidth=1)
            ax.set_xlabel('Routing Steps / Layers')
            
        ax.set_title("Mixture-Of-Experts Workload Balance")
        ax.set_ylabel('Activation Utilization Frequency')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Puts legend outside
        plt.tight_layout()
        plt.show()
        
    def plot_router(
    self,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    max_tokens: Optional[int] = None,
    ):
        """
        Visualize router probabilities for one MoE layer.

        Rows    = tokens
        Columns = experts
        Values  = router probability
        """
        import matplotlib.pyplot as plt
        import numpy as np

        self.eval()

        with torch.no_grad():
            _ = self(input_ids)

        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(
                f"layer_idx must be between "
                f"0 and {len(self.layers) - 1}"
            )
        moe = self.layers[layer_idx].block.moe
        if moe.last_router_probs is None:
            raise RuntimeError(
                "Router telemetry was not collected."
            )
        router_probs = moe.last_router_probs.numpy()

        # Currently the telemetry is flattened:
        #
        # [batch * sequence, experts]
        #
        # For visualization, select the first sequence.
        batch, sequence = input_ids.shape
        router_probs = router_probs.reshape(
            batch,
            sequence,
            -1,
        )
        probabilities = router_probs[0]
        if max_tokens is not None:
            probabilities = probabilities[:max_tokens]

        # --------------------------------------------------------------
        # Token labels
        # --------------------------------------------------------------
        
        tokens = input_ids[0].detach().cpu().tolist()

        if max_tokens is not None:
            tokens = tokens[:max_tokens]

        labels = [
            str(token)
            for token in tokens
        ]
        # --------------------------------------------------------------
        # Plot
        # --------------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )
        image = ax.imshow(
            probabilities.T,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Expert")
        ax.set_title(
            f"MoE Router Probabilities — Layer {layer_idx}"
        )
        ax.set_yticks(
            np.arange(probabilities.shape[-1])
        )
        ax.set_yticklabels(
            [
                f"Expert {i}"
                for i in range(probabilities.shape[-1])
            ]
        )
        ax.set_xticks(
            np.arange(len(labels))
        )
        ax.set_xticklabels(
            labels,
            rotation=90,
        )
        fig.colorbar(
            image,
            ax=ax,
            label="Router Probability",
        )
        plt.tight_layout()
        plt.show()

    def residual_report(self) -> None:

        print()
        print("=" * 60)
        print("HyperConnection Residual Report")
        print("=" * 60)

        for layer_idx, layer in enumerate(self.layers):

            alpha = (
                layer.alpha.detach()
                .cpu()
                .item()
            )

            print(
                f"Layer {layer_idx:02d} | "
                f"alpha={alpha:.6f}"
            )

        print("=" * 60)

    def router_report(
    self,
    layer_idx: int = 0,
    ) -> None:
        """
        Print a compact analysis of one MoE router.
        """

        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError("Invalid layer index")

        moe = self.layers[layer_idx].block.moe

        if moe.last_router_probs is None:
            raise RuntimeError(
                "Run a forward pass before requesting router diagnostics."
            )

        probs = moe.last_router_probs

        top_experts = probs.argmax(dim=-1)

        print()
        print("=" * 60)
        print(f"MoE Router Report — Layer {layer_idx}")
        print("=" * 60)

        for expert_id in range(probs.shape[-1]):

            mean_probability = probs[:, expert_id].mean().item()

            selection_frequency = (
                top_experts == expert_id
            ).float().mean().item()

            print(
                f"Expert {expert_id}: "
                f"mean probability={mean_probability:.4f} | "
                f"top-1 frequency={selection_frequency:.2%}"
            )

        print("=" * 60)

    def expert_health_report(self):

        print()
        print("=" * 70)
        print("Expert Health Report")
        print("=" * 70)

        for layer_idx, layer in enumerate(self.layers):

            usage = layer.block.moe.last_usage

            if usage is None:
                continue

            print(f"\nLayer {layer_idx}")

            for expert_idx, value in enumerate(usage):

                value = value.item()

                if value < 0.01:
                    status = "⚠ DEAD / RARELY USED"

                elif value > 0.50:
                    status = "⚠ DOMINANT"

                else:
                    status = "✓ ACTIVE"

                print(
                    f"  Expert {expert_idx}: "
                    f"{value:.2%}  {status}"
                )

        print("=" * 70)

    def gradient_report(
    self,
    sort_by: str = "name",
    top_n: Optional[int] = None,
    ):
        """
        Analyze gradients after loss.backward().

        Must be called AFTER:
            loss.backward()

        Reports:
            mean
            std
            min
            max
            L2 norm
            percentage of zero values
            NaN / Inf detection
        """

        rows = []

        for name, parameter in self.named_parameters():

            if parameter.grad is None:
                rows.append({
                    "name": name,
                    "status": "NO_GRAD",
                })
                continue

            grad = parameter.grad.detach().float()

            rows.append({
                "name": name,
                "mean": grad.mean().item(),
                "std": grad.std().item()
                    if grad.numel() > 1
                    else 0.0,
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
                "zero_pct": (
                    (grad == 0).float().mean().item()
                    * 100
                ),
                "nan": torch.isnan(grad).any().item(),
                "inf": torch.isinf(grad).any().item(),
            })

        # --------------------------------------------------------------
        # Sorting
        # --------------------------------------------------------------

        if sort_by == "norm":
            rows = sorted(
                rows,
                key=lambda row: row.get("norm", -1),
                reverse=True,
            )

        elif sort_by == "name":
            rows = sorted(
                rows,
                key=lambda row: row["name"],
            )

        if top_n is not None:
            rows = rows[:top_n]

        # --------------------------------------------------------------
        # Display
        # --------------------------------------------------------------

        print()
        print("=" * 120)
        print("Gradient Diagnostics")
        print("=" * 120)

        header = (
            f"{'Parameter':<45}"
            f"{'Mean':>12}"
            f"{'Std':>12}"
            f"{'Norm':>12}"
            f"{'Zero %':>10}"
            f"{'Status':>15}"
        )

        print(header)
        print("-" * 120)

        for row in rows:

            if row["status"] == "NO_GRAD":

                print(
                    f"{row['name']:<45}"
                    f"{'':>12}"
                    f"{'':>12}"
                    f"{'':>12}"
                    f"{'':>10}"
                    f"{'NO_GRAD':>15}"
                )

                continue

            status = "OK"

            if row["nan"]:
                status = "NaN"

            elif row["inf"]:
                status = "INF"

            elif row["norm"] < 1e-8:
                status = "VANISHING"

            elif row["norm"] > 1e2:
                status = "LARGE"

            print(
                f"{row['name']:<45}"
                f"{row['mean']:>12.3e}"
                f"{row['std']:>12.3e}"
                f"{row['norm']:>12.3e}"
                f"{row['zero_pct']:>9.2f}%"
                f"{status:>15}"
            )

        print("=" * 120)

        return rows

    def gradient_summary(self) -> dict[str, float]:

        total_norm_sq = 0.0
        parameter_count = 0
        parameters_with_grad = 0
        parameters_without_grad = 0
        nan_parameters = 0
        inf_parameters = 0

        for parameter in self.parameters():

            parameter_count += 1

            if parameter.grad is None:
                parameters_without_grad += 1
                continue

            parameters_with_grad += 1

            grad = parameter.grad.detach()

            if torch.isnan(grad).any():
                nan_parameters += 1

            if torch.isinf(grad).any():
                inf_parameters += 1

            total_norm_sq += (
                grad.float().norm().item() ** 2
            )

        total_norm = total_norm_sq ** 0.5

        summary = {
            "total_gradient_norm": total_norm,
            "parameter_count": parameter_count,
            "parameters_with_grad": parameters_with_grad,
            "parameters_without_grad": parameters_without_grad,
            "nan_parameters": nan_parameters,
            "inf_parameters": inf_parameters,
        }

        print()
        print("=" * 60)
        print("Global Gradient Summary")
        print("=" * 60)

        for key, value in summary.items():
            print(f"{key:<30}: {value}")

        print("=" * 60)

        return summary

        
    def forward(self, input_ids: torch.Tensor, token_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.padding_aware_routing and token_mask is None:
            token_mask = input_ids != 0
        return self.forward_embeddings(self.embedding(input_ids), token_mask=token_mask) # Input_ids get turned into embeddings, which get forward passed through the model layers
    
    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        routed_total = sum(
            sum(p.numel() for expert in layer.block.moe.experts for p in expert.parameters()) # Sums up the parameters of all individual expert sub-networks across all layers, isolating how much of the model's total capacity belongs to the specialized "expert" pools
            for layer in self.layers
        )
        training_only = sum(p.numel() for p in self.future_predictors.parameters())
        active = total - training_only - routed_total + routed_total * self.config.top_k // self.config.experts # Calculates how many parameters are actually utilized to process a single token: routed - routed_total removes all expert parameters from the total, leaving only the "dense backbone" layers (e.g. Embeddings, RMSNorms, Linear/Sparse Attention, and HyperConnection logits) which are always executed for every token | routed_total * self.config.top_k // self.config.experts calculates the fractional size of the activated experts
        return {
            'total': total, 'active_per_token_estimate': active,
            'training_only_parameters': training_only, 'inference_parameters': total - training_only
        }

    def router_losses(self):
        """Differentiable auxiliary terms from the most recent forward."""
        return (torch.stack([layer.block.moe.aux_loss for layer in self.layers]).mean(),
                torch.stack([layer.block.moe.z_loss for layer in self.layers]).mean())

    def update_router_biases(self):
        for layer in self.layers:
            layer.block.moe.update_bias()

    def future_logits(self):
        """Additional byte-offset heads from the latest forward; ordinary generation stays next-byte."""
        if self._future_hidden is None:
            return []
        values = [self.output(head(self._future_hidden)) for head in self.future_predictors]
        if self.config.logit_softcap:
            cap = self.config.logit_softcap
            values = [cap * torch.tanh(v / cap) for v in values]
        return values

    def future_prediction_loss(self, labels):
        from .objectives import future_byte_loss
        return future_byte_loss(self.future_logits(), labels, self.embedding.weight)

    def parameter_counts(self):
        return self.param_count()


# Keep Vuk's imports working while exposing the modded project name.
GLM53FlashFromScratch = ModdedGLM53FlashFromScratch
SparseMoE = SparseMixtureOfExperts
apply_rope = apply_rot
