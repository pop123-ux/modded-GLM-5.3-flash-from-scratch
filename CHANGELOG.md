# 0.3.0 — Five optional research extensions

- Real-token expert dispatch, masking-aware router losses and explicit right-padding crop.
- Chunked/checkpointed causal linear attention with compact differentiable boundary states.
- Dense SwiGLU controls sized by active expert projection width.
- Explicit, bounded, non-gradient expert selection-bias updates with checkpoint persistence.
- Auxiliary future-byte heads, shifted masks and separate training-only parameter counts.
- 22 experiment arms, six new presets, saved-storage profiling, community contribution templates, and 20 ranked research ideas.
- Extension correctness tests verify numerical/gradient equivalence, masking, controller behavior, label offsets and checkpoint round trips.

All model additions default off. No GPU speedup or convergence benefit is asserted.

# 0.2.0 — Repaired modded experiment lab

## Preserved from the uploaded work

Annotated architecture and tensor tracing; 8-stream residual design; learned alpha and dropout; 13 coding families; byte tokenizer and pair encoding; educational notebooks. Original annotated sources are also retained verbatim as text notes.

## Repairs

- Consistent top_k/streams/head_dim API, even RoPE dimensions, early validation.
- Correct RMSNorm multiplication, valid stream expansion, initialized trace flags.
- Functional runtime and evaluator, populated opt-in stream/router diagnostics.
- Eliminated duplicate sparse QKV projection; cached sparse indices and rotary tables.
- Corrected exp(2.08) reference value; used float tolerance for floating tasks.
- Restored MIT attribution; marked formerly empty notebooks as unfinished valid notebooks.

## New experiment infrastructure

- Configurable gathered/masked/full attention, QK norm, embedding skips, logit softcap and standard residual path.
- Differentiable balancing and router z-loss; portable Muon-style plus AdamW optimizer.
- Full/completion label masks; token-weighted validation; synchronized timing and data/source receipts.
- Checkpoint inference compatibility with both active and pinned models; saved optimizer state (no automatic resume).
- 17-condition paired ablation runner, profiler, benchmark analysis and preset configs.
- Expanded correctness tests including upstream equivalence and policy-gradient scoring.

## Upstream workflows adapted

Pretraining, RLOO, greedy/pass@k evaluation, generation/runtime, result analysis, system receipts, miniature vision, full-width digit pilot and curriculum/data-diversity studies. Active vision is completed using the upstream implementation; incomplete source notes are retained separately.

No GPU record, convergence win, statistical significance claim or trained model release is asserted.
