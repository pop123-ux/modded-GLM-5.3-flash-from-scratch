# Architecture and implementation

## Model path

Byte token IDs → tied embedding → residual streams → repeated attention/MoE blocks → mean over streams → RMSNorm → tied vocabulary projection.

The hybrid mode uses positive-feature causal linear attention in three blocks and gathered sparse attention in each fourth block by default. Routing selects top-k experts and adds one always-active shared expert. Experts use SwiGLU. This is a readable teaching model, not the production GLM attention or routing algorithm.

The eight-stream mechanism uses global learned softmax mixing/routing vectors, optional dropout and a learned update scale. It does not implement manifold-constrained hyper-connections. Setting `residual_mode="standard"` uses the block's ordinary residual updates and bypasses stream routing. Unused routing/scaling parameters still exist in that ablation, so count reports are architectural estimates rather than exact FLOP measures.

## Corrections

Config fields are now `top_k`, `streams`, and a derived `head_dim`; read-only `topk` and `num_streams` aliases remain for notebooks. Defaults use 128 dimensions/4 heads, yielding an even 32-dimensional RoPE head. Invalid dimensions/routing/backend choices fail during configuration construction.

RMSNorm multiplies by reciprocal root mean square. Stream expansion uses `expand`, not negative repeat counts. Diagnostics are explicitly opt-in. Stream norms/similarities and router probabilities are populated only when requested. Original annotated sources are preserved as text under `theory/user_source_notes/`, including obsolete code for reference; they are not the runnable implementation.

## Attention options

`attention_mode="hybrid"` retains the hybrid model. `sparse_backend="gather"` gathers actual allowed keys/values. `sparse_backend="sdpa"` applies precisely the same local-plus-strided mask using PyTorch SDPA. Tests compare outputs and input gradients between these paths.

`attention_mode="sdpa"` changes every block to full causal SDPA; this changes the architecture. The profiler cannot establish which architecture reaches a quality target faster.

The default linear attention implementation materializes prefix outer products `[B,T,H,d,d]`; it has substantial activation memory. An optional chunked/checkpointed implementation carries differentiable compact boundary states and trades recomputation for smaller saved activations. Fixed-stride anchors grow with history, so gathered attention has approximately O(T*window + T²/stride) score work, not asymptotically linear cost. The masked SDPA path explicitly allocates a dense T×T mask. Kernel availability/performance varies by device and PyTorch version; “SDPA” does not guarantee FlashAttention.

Rotary and sparse caches are bounded to the most recent shape/device/dtype. They are not checkpoint state and rebuild automatically after device/shape changes.

## Router losses

For E experts, average full router probabilities p and detached hard assignment fractions f:

`balance_loss = E * sum(f * p)`

Assignments are normalized across all selected slots. At equal usage the loss is approximately 1, not 0. Gradients flow through p, including probabilities of unselected experts. The router z-loss is the mean square log-sum-exp of router logits. Both are averaged over layers and their weights default to zero in the pretraining CLI so experiments can isolate them.

Default routing retains all input positions for compatibility. Opt-in padding-aware routing excludes masked inputs from dispatch, usage, probability averages and z-loss. Attention still assumes right-padded independent rows; the routing mask is not a document-attention mask. See research/EXTENSIONS.md for details.

## Optimizer

The optional Muon-style optimizer applies Nesterov momentum, a five-step quintic Newton–Schulz transform, aspect-ratio scaling and decoupled decay to hidden matrices. Embeddings/tied output, routers and nonmatrix parameters use AdamW. The implementation computes the transform in FP32 for portability; it is not the latest sharded/BF16 NanoGPT implementation and is not proven faster here. LR for Muon is separately tunable via `--muon-lr`.

Model weights/config/metadata support inference round trips. Optimizer states are also saved after requested training checkpoints; automatic resume is not implemented. A new training invocation always starts from random initialization. RL explicitly starts from a supplied model checkpoint. Rollouts and policy scoring both disable dropout; eval mode does not disable gradient recording.

## Data and evaluation

The corpus retains your 13 coding families. Full-sequence pretraining and completion-only loss are selectable; receipts identify the choice. Validation aggregates summed cross entropy over actual unmasked tokens rather than averaging batch means. Both the selected primary loss and completion-only loss are reported.

Frozen splits have different names/seeds but share task families and completion templates. This evaluates template learning and naming robustness, not general programming ability. Random names do not eliminate template overlap. Curriculum/diversity studies additionally use compositions of arithmetic operations with held-out structures, as described in their scripts.

Executable rewards allow only a restricted Python subset plus `math.exp`/`math.factorial`. Floating references use close comparison; other cases require type and value equality. Each candidate is verified before being run in a subprocess with a wall-clock timeout and POSIX resource limits where available. This is not an OS sandbox and is intended for locally generated examples, not arbitrary hostile submissions.

Generation stops at EOS or the token cap. It no longer stops at the first parseable prefix, which could truncate a multi-branch function. Caps of 48 bytes are too short for some families; use `--max-new-tokens 192` for all-family evaluations when context allows.

## Vision and compatibility

The completed vision path is adapted from the pinned upstream implementation: RGB patch embedding, bidirectional transformer, 2×2 merger, multimodal projector and placeholder replacement. User-facing class aliases are retained. `train_vision.py` compares it with direct patch projection on procedural digits. The larger digit pilot uses the original ~25.7M dimensions plus one residual scalar per layer; arbitrary compatible modded checkpoints can be expanded to 263-token vocabulary.

The exact upstream files are under `baselines/upstream/`; executable active copies include compatibility fixes and are not claimed to be unmodified. Do not run pytest recursively across both trees: `python -m pytest` targets only the active `tests/` directory.

## Research options

See [research/EXTENSIONS.md](research/EXTENSIONS.md) for dense expert-width controls, padding-aware dispatch, chunked attention, explicit bias updates and auxiliary future-byte heads. Model flags default off, preserving existing configurations. Training-only head parameters are excluded from the inference active-parameter estimate.
