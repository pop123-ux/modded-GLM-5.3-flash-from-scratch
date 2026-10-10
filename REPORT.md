# Verification report — v0.3.0 — 2026-10-10

The expanded project passes **76 tests** on CPU. All **22 ablation conditions** completed three-step tiny-model training runs with one model seed (42), fixed data seed, batch size 2 and 320-byte maximum context. All recorded losses and gradient norms are finite. These runs verify functionality; they do not establish convergence, coding ability or speedrun improvements.

## Current verification

- Tests cover causal outputs, checkpoint round trips, pinned upstream equivalence, routing losses, rollout gradients, tokenizer/data behavior and the five research additions.
- Chunked linear attention agrees with the prefix reference in outputs, input gradients and layer parameter gradients. Chunk sizes and checkpoint modes are covered. Boundary states own compact storage rather than retaining full chunk views.
- Masked routing agrees with compact real-token routing; padded inputs receive no expert gradients. Right-padding cropping retains supervised targets and primary validation loss in the tested settings.
- Dense control has the intended active expert matrix width and finite gradients.
- Bias control changes only after the explicit training update, consumes usage once, and remains frozen in evaluation. Selection bias does not replace mixture-weight logits.
- Future-byte objectives use tested offsets and ignore padding. Gradients reach the auxiliary heads and shared trunk. Training-only parameters are reported separately.
- All five new arms and the hybrid control were rerun after the final attention-state fix. The four compatible MoE additions also completed a combined three-step run; checkpoint reload and independent frozen dev evaluation succeeded. Dense FFNs are a separate control.
- Python compilation and shell syntax checks completed. Linux and Windows CI definitions include the test suite and two training smoke runs; hosted CI was not executed here.

## Isolated attention-storage measurement

For CPU float32 features shaped B=1, T=128, H=2, d=8, chunk size 16, saved-tensor hooks counted **108,544 bytes** of unique storage referenced by the prefix reference versus **29,184 bytes** by checkpointed chunks. Both backward passes had finite gradients. This includes inputs and deduplicates shared storage; it is **not peak allocator memory or full-model memory**, and does not measure speed. Reproduce with `experiments/profile_linear_memory.py`.

## Evidence and limits

Current receipts and logs are in `artifacts/research-smoke/`; test output and XML are in `artifacts/`. No trained weights are packaged. The earlier release's broader smoke checks and report are preserved under `artifacts/cpu-smoke/` and `artifacts/PREVIOUS_REPORT_v0.2.md`; those are historical checks, not reruns of every CLI for v0.3.0.

Environment: Linux x86_64, Python 3.12.14, PyTorch 2.5.1+cpu. CUDA, MPS, BF16, multi-GPU execution, Windows execution and longer multi-seed experiments remain unverified. A GPU leaderboard is intentionally empty. New features are disabled in the default model and have individual presets for controlled comparisons.

The default model still has 12,625,740 stored parameters and an architectural active-per-token estimate of 7,907,148. Future heads add training-only parameters when enabled. Neither count is a hardware FLOP measurement.

This is suitable for publication as an educational research project with these limits disclosed. Public speedrun records should follow BENCHMARK_RULES.md and CONTRIBUTING.md and include matched hardware, complete receipts, confirmation evaluation and multiple seeds.
