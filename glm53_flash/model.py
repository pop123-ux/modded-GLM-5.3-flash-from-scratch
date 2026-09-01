"""GLM-Flash-style-like model

This is not a 1:1 reproduction of GLM-5.3-Flash.
It was made purely for educational purposes and learning some of the novel AI concepts: MoE, Residual Streams, hybrid-attention etc.
It was also made in order to go in more depth regarding some basic python concepts such as annotations, dataclasses, repository structure & etiquette etc.
"""
from dataclasses import asdict, dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True) # The python dataclass is a built-in tool that automaticall generates boilerplate methods for data-storing classes. The frozen=True argument makes the dataclass immutable, preventing fields from being modified after creation and making the object safely hashable
class ModelConfig:
    """Standard Transformer Architecture Components"""
    vocab_size: int = 256 + 4 * 1 # Using byte-level tokenization we need 256 starter tokens + <bos> + <eos> + <pad> + <unk>
    dim: int = 126 # The size or length of a vector used to represent data | In our case, every single token is represented by a vector of 128 numbers | Scaling of the dim variable allows the model to capture more complex relationships between words | Observation: Every time we double the dim, the amount of memory and compute required by the model's linear layers increases close to quadratically (x**2) | Essentially the width of the model
    layers: int = 12 # The total number of model layers | Transform raw input tokens into complex concepts | Essentially the depth of the model
    heads: int = 6 # Sub-units within a single layer that split the model's total dimension to look at a sentence from multiple perspectives at the same time -> Multi-Head Attention | Dimension per head -> head_dim: int = dim / heads = 32 (size of each individual head) A single vector of 192 numbers is sliced into 6 smaller vectors of 32 numbers.
    """Mixture-Of-Experts Components"""
    expert_hidden: int = 4 * 126 # 4 x dim (hidden_size) = 512 | Expert number of neurons - part of MoE architecture
    experts: int = 4 # Number of experts used - also part of the MoE architecture
    topk: int = 2 # Top 2 Mixture of Experts (MoE) routing - the exact number of expert networks a single token will be sent to at every MoE layer.
    """Residual-Streams Component"""
    num_streams: int = 8 # Manifold-Constrained Hyper-Connections (https://arxiv.org/pdf/2601.02451) | Enables the raw internal matrices to transform the standard layout into a much larger multidimensional tensor [batch_size, sequence_length, number_of_streams, dim]
    """Block Sparse-Attention Components"""
    sparse_window: int = 16 # The number of continuous, back-to-back tokens the model can look at during a single hop | Enables the model to capture local context
    sparse_stride: int = 16 # Dictates the step size across the sequence history | If sparse_stride: 16, the model skips over 16 tokens before activating its next attention window | Allows the model to look far back into its past memory without causing the GPU to crash
    """Inference"""
    max_sequence_length: int = 126 # Absolute maximum number of tokens a model can process or generate in a single prompt session
    
    # ------------------------------------------------------------------
    # Educational / Diagnostics
    # ------------------------------------------------------------------
    trace_shapes: bool = False
    
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
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # Essentially the gamma from the outputs normalization using RMSNorm formula
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / scale).to(x.dtype) * self.weight

class LinearAttention(nn.Module):
    """Causal positive-feature linear attention with recurrent prefix state."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(in_features=config.dim, out_features=config.dim * 3, bias=False) # [batch_size, seq_len, config.dim] -> [batch_size, seq_len, config.dim * 3]
        self.out = nn.Linear(in_features=config.dim, out_features=config.dim, bias=False) # Final projection layer of the attention block, often called the output projection/dense layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        """Reshaping Q, K, V tensors to respect the Multi-Head Attention Layout."""
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rot(q, k)
        q = F.elu(q) + 1.0 # The Exponential Linear Unit (ELU) activation function outputs values down to -1.0. By adding 1.0, we force every single number in q and k to be >= 0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bthd,bthe->bthde", k, v).cumsum(dim=1) # Shape [batch, length, heads, head_dim_k, head_dim_v] Computes the KV-cache and enforces causality (tokens can only look at past tokens, not future ones)
        k_prefix = k.cumsum(dim=1) # Shape [batch, length, heads, head_dim_k] Computes the cumulative sum of the Key tensor alone along the Time axis | Storing the running sum of all past keys allows us to track the total "weight" of the history up to the current token
        n = torch.einsum('bthd,bthde->bthe', q, kv) # Shape [batch, length, heads, head_dim_v] (this line basically computes: [head_dim_k] * [head_dim_k, head_dim_v] = [head_dim_v]) Multiplies the Q (Query) by the running historical memory matrix (kv)
        d = torch.einsum('bthd,bthd->bth', q, k_prefix).unsqueeze(-1) # Shape [batch, length, heads, 1] | Computes dot product between the current Q (Query) and the accumulated historical Keys (k_prefix) at each time step | Acts as the replacement for the Softmax denominator, scaling the features so that tokens don't explode in value as the text gets longer
        """Complete the causal attention lookup in a perfectly linear time-frame"""
        output = n / d.clamp_min(1e-6) # Shape [batch, length, self.heads, self.head_dim] | Prevents division by 0 error
        return self.out(output.reshape(batch, length, dim))

class SparseAttention(nn.Module):
    """Gathered causal attention over a local window plus strided anchors."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.window = config.sparse_window
        self.stride = config.sparse_stride
        self.qkv = nn.Linear(in_features=config.dim, out_features=config.dim * 3, bias=False)
        self.out = nn.Linear(in_features=config.dim, out_features=config.dim, bias=False)
    
    """Method that inputs the total sequence length (length) and the destination hardware, returning a tuple of 2 tensors: one for row indices and one for column indices"""
    def _indices(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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
        return indices, valid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        q, k = apply_rot(q, k)
        q = q.transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        k = k.transpose(1, 2) # Shape [batch, self.heads, length, self.head_dim]
        indices, valid = self._indices(length, x.device) # self._indices() produces Shape [length, k]
        gathered_k = k[:, :, indices, :] # Shape [batch, self.heads, length, k, self.head_dim]
        gathered_v = v[:, :, indices, :] # Shape [batch, self.heads, length, k, self.head_dim]
        scores = torch.einsum('bhtd,bhtkd->bhtk', q, gathered_k) * self.head_dim**-0.5 # Shape [batch, self.heads, length, k] | torch.einsum(...) does the dot product between q and gathered_k | * self.head_dim **-0.5 does the scaling -> Scaled Dot-Product in order to prevent the gradients from exploding
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min) # Masks the attention scores (logits) by setting padded or invalid positions to -infinity, ensuring they receive exactly 0% attention weight after the Softmax operation
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype) # ransforms the attention logits into probability weights (0.0 to 1.0)
        outputs = torch.einsum('bhtk,bhtkd->bhtd', weights, gathered_v) # Shape [batch, self.heads, length, self.head_dim]
        return self.out(outputs.transpose(1, 2).reshape(batch, length, dim)) # Shape [batch, length, dim = heads * head_dim]
    
class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(in_features=dim, out_features=hidden * 2, bias=False)
        self.down = nn.Linear(in_features=hidden, out_features=dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)
    
""" Visual representation of the Expert class

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
        
    def forward(self, x: torch.Tensor, return_usage: Optional[bool] = True) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape # Shape of input
        flat = x.reshape(-1, original_shape[-1]) # Reshapes multi-dimensional input tensor into a 2D tensor | e.g: ([32, 3, 224, 224] -> [32*3*224, 224])
        logits = self.router(flat) # Routes the input to experts to receive the tensor of logits of each
        top_values, top_indices = logits.topk(self.top_k, dim=-1) # Returns the values and indices of top_2 highest active experts given the input
        top_weights = torch.softmax(top_values.float(), dim=-1).to(flat.dtype) # Returns the probabilities
        output = self.shared(flat) # Passes all tokens through a self.shared expert layer
        usage = torch.zeros(len(self.experts), device=x.device, dtype=torch.float32) # Counts how many tokens are assigned to each expert
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_id) # Looks through the top choices of all tokens to see which ones selected the current expert_id | positions specifies which tokens chose this expert; slots specifies whether it was the 1st choice or 2nd choice (up to 2 in our case)
            if positions.numel() == 0: # If no tokens selected the specific expert in this batch, it skips the rest of the loop for this expert
                continue
            contribution = expert(flat[positions]) # Extracts only the specific tokens assigned to this expert via flat[positions] and passes them through the current expert neural network layer
            contribution = contribution * top_weights[positions, slots, None].to(contribution.dtype) # Fetches the probabilities corresponding to the specific token assignments and multiplies them by the expert's output
            output.index_add_(0, positions, contribution) # Adds the scaled expert results back into the main output tensor at the specific row indices indicated by positions | index_add_ accumulates the contributions from both experts into the same token slot, overlaying the results on top of the shared expert output initialized earlier
            usage[expert_id] = positions.numel() # Counts the total number of tokens processes by this expert and stores it in the usage tracker array
        usage = usage / max(1, flat.shape[0] * self.top_k) # Normalizes the workload counts into a percentage (0-1) representing the fraction of total expert slots this expert processed | flat.shape[0] * self.top_k -> total number of tokens multiplied by K | If a few experts have a usage of 1.0 while others have 0.0, it means the model has bottlenecked by "expert collapse" (not utilizing its full capacity) | max(1, ...) acts as a safegoard against zero-division errors
        
        if return_usage:
            return output.view(original_shape), usage # New shape [batch_size, self.length, self.hidden_dim] Reshapes the flattened 2D output matrix back to its exact initial multi-dimensional structure
        else:
            return output.view(original_shape)
            

class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = SparseAttention(config) if sparse else LinearAttention(config)
        self.moe = SparseMixtureOfExperts(config)
    
    """Pre-layer normalization and standard residual (skip) connections across two primary processing phases"""
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x)) # Phase 1: Attention Processing | x + self.attention(...) represents a Residual Connectioon that prevents vanishing gradients during backpropagation
        ffn, usage = self.moe(self.ffn_norm(x)) # Phase 2: Feed-Forward Processing (MoE) | The updated tensor x is normalized again using self.ffn_norm, which is then passed to the MoE layer, which returns ffn: the actual processed features output by the selected active experts; usage: a structural tensor tracking how frequently individual experts were chosen
        return x + ffn, usage
    
class HyperConnection(nn.Module):
    """Eight residual streams w/ learned input mixing and output routing."""
    
    def __init__(self, config: ModelConfig, block: HybridBlock):
        super().__init__()
        self.block = block
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams)) # Tensor representing the starter input logits evenly spaced from 0.1 to -0.1 with a size of 8 | nn.Parameter means that PyTorch will adjust the values during backpropagation
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams)) # Tensor representing the starter output logits evenly spaced from 0.1 to -0.1 with a size of 8 | nn.Parameter means that PyTorch will adjust the values during backpropagation
    
        self.alpha = nn.Parameter(torch.tensor(0.01)) # Scalar initialized initially with 0.01, it reduces the influence of the block so that the network will perform similar to a clean residual connection
        self.dropout = nn.Dropout(0.1) # Randomly zeroes out 10% of the generated features during training, forcing the 8 residual streams to learn independent representations
    
    """Residual streams forward pass, expects an input tensor streams of shape [Batch, Tokens, Streams, Dim]"""
    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = torch.einsum(
            "s,btsd->btd", torch.softmax(self.input_logits, dim=0), streams # torch.softmax(..., dim=0) normalizes the input logits into a valid probability distribution | torch.einsum performs weighted average, multiplying each stream weight s by its corresponding features in streams (btsd), basically mixing the information acquired
        )
        transformed, usage = self.block(mixed) # Passes the mixed representation tensor directly into HybridBlock, processing the combined token embeddings through Attention & MoE, returning the updated values (transformed) and the structural expert allocation matrix (usage)
        routed = torch.softmax(self.output_logits, dim=0) # Converts the learnable output logits into a probability distribution, deciding what percentage of the processed output vector gets distributed to each of the 8 streams
        features = transformed.unsqueeze(2) * routed[None, None, :, None] # Final shape [Batch, Tokens, Streams, Dim] transformed.unsqueeze(2) -- [B, T, D] -> [B, T, 1, D] | routed[None, None, :, None] reshapes the 1D routing vector into a [1, 1, S, 1] tensor 
        streams = streams + self.alpha * self.dropout(features) # Applies dropout to the generated stream additions, scales them by the learnable self.alpha barrier factor, and adds them to the original input tensor via a classic residual skip connection
        return streams, usage # Returns the newly updated streams tensor alongside the expert usage tensor received from the MoE layer
        
class ModdedGLM53FlashFromScratch(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.heads: # This safety check guarantees that the hidden dimension (config.dim) can be perfectly split across the multi-head attention mechanisms (config.heads) | If it is not divisible, Multi-Head Attention cannot evenly partition vector spaces, which throws runtime errors
            raise ValueError("dim must be divisible by heads")
        self.config = config 
        self.embedding = nn.Embedding(config.vocab_size, config.dim) # Maps raw integer token IDs into continuous dense vector representations of size config.dim, preparing them for neural processing
        self.layers = nn.ModuleList([
            HyperConnection(config, HybridBlock(config, sparse=((index + 1) % 4 == 0))) # Instantiates the model's structural depth (config.layers) | by using ((index + 1) % 4 == 0), every 4th layer is initializes with a SparseAttention mechanisms inside of its HybridBlock | The remaining 3 intermediate layers use LinearAttention, mimicking structural techniques found in state-of-the-art models like GLM-5.3-Flash to balance long-context computational costs | Each HybridBlock is wrapped inside a HyperConnection instance to enforce multi-stream routing and prevent representation decay across the layers
            for index in range(config.layers)
        ])
        self.final_norm = RMSNorm(config.dim) # Before generating vocabulary predictions, the hidden states pass through a final Root Mean Square Normalization layer to standardize data magnitudes across the batch
        self.output = nn.Linear(in_features=config.dim, out_features=config.vocab_size, bias=False) # This linear projection shifts the output dimension from the model's internal hidden space back to a massive array matching the vocabulary length, generating raw logit scores for every possible token | 192 -> 260
        self.output.weight = self.embedding.weight # This memory optimization trick shares the parameter weights between the input embedding layer and the final output prediction layer, cutting down overall model parameter count significantly, enforcing semantic alignment (a token's representation when reading text matches its representation when generating text)
        self.apply(self._init_weights) # Recursively sweep through every submodule defined in this class and apply a custom weight initialization routine to ensure parameters start with optimal distributions before training
        
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # Fills module.weight with values drawn from the normal distribution, with a mean of 0 and custom std of 0.02
            
    def forward_embeddings(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the language model from precomputed token-like embeddings."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded inputs must have shape [batch, time, dim] (B, T, D)")
        if embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("embedded input exceeds max_sequence_length")
        streams = embedded.unsqueeze(2).repeat(-1, -1, self.config.streams, -1) # Multi-Stream Setup; New shape [B, T, 8, D] | embedded.unsqueeze(2) transforms [B, T, D] -> [B, T, 1, D] | .expand(-1, -1, self.config.streams, -1) copies the same entry embeddings from each 8 parallel streams | It allocates memory
        usages = [] # Empty list to collect the usage logs generated by each SparseMixtureOfExperts routing layer
        for layer in self.layers: # Loops through the stacked layer infrastructure | Each layer (HyperConnections instance) takes the [B, T, 8, D] streams tensor, passes a compressed version through its inner HybridBlock, updates the streams, and returns the newly modified streams along with a usage tensor tracking expert activation
            streams, usage = layer(streams)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2)) # Compresses the tensor down to [Batch, Time, Dim], collapsing the multi-stream dimension by averaging the 8 lanes, combining all independently learned structural & semantic properties into a single unified context vector
        return self.output(hidden), torch.stack(usages) # self.output projects the hidden dimension (D) up to the full vocab size (Vocab), yielding a final prediction tensor of shape [Batch, Time, Vocab] containing raw token generation logits | torch.stack(usages) converts the list of separate layer metrics into a single, cohesive PyTorch tensor, making it incredibly easy to calculate the load-balancing loss during training to ensure no experts are left idle
    
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
        
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embedding(input_ids)) # Input_ids get turned into embeddings, which get forward passed through the model layers
    
    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        routed_total = sum(
            sum(p.numel() for expert in layer.block.moe.experts for p in expert.parameters()) # Sums up the parameters of all individual expert sub-networks across all layers, isolating how much of the model's total capacity belongs to the specialized "expert" pools
            for layer in self.layers
        )
        active = total - routed_total + routed_total * self.config.top_k // self.config.experts # Calculates how many parameters are actually utilized to process a single token: routed - routed_total removes all expert parameters from the total, leaving only the "dense backbone" layers (e.g. Embeddings, RMSNorms, Linear/Sparse Attention, and HyperConnection logits) which are always executed for every token | routed_total * self.config.top_k // self.config.experts calculates the fractional size of the activated experts
        return {
            'total': total, 'active_per_token_estimate': active
        }