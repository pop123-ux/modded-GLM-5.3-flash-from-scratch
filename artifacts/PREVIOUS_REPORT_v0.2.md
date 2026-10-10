# Verification report — 2026-10-10

## Result

The repaired project is runnable on CPU, with **44 passing tests**. All **17 ablation conditions** completed three-step tiny-model training runs (one model seed, fixed data seed, batch size 2, 320-byte context). This is functional verification, not convergence or speedup evidence.

Environment: Linux x86_64, Python 3.12.14, PyTorch 2.5.1+cpu. CUDA/MPS, Windows, BF16 and the added GitHub Actions workflow were not executed here. The workflow is provided for your repository.

## Executed checks

- Entire active pytest suite: 44 passed; XML and console result included.
- All 17 tiny-model ablation arms: Vuk, hybrid, repaired user design, small alpha, dropout, stream counts 1/4, standard residual, sparse SDPA, full SDPA, QK norm, embedding skips, softcap, balancing, z-loss, Muon and cosine schedule.
- Default 12-layer / 128-wide model: one CPU optimizer step plus validation on increment examples.
- Completion-only training with Muon, balancing, z-loss and target-loss tracking: two steps, successful receipt.
- Checkpoint round trips for both active and pinned implementations, including output equality.
- Pinned Vuk versus repaired hybrid equivalence after copying weights, with alpha=1 and dropout=0.
- Sparse gather versus equivalent masked SDPA output/input-gradient agreement.
- Changed future tokens cannot alter past logits across attention/residual variants.
- Router auxiliary losses backpropagate; gradients are finite.
- Separate RLOO policy-gradient test verifies differentiable rollout scoring and leave-one-out advantages.
- RLOO CLI: two rollout groups. All candidates were invalid and rewards equal, so no update occurred. This is an expected smoke result, not RL improvement.
- Greedy evaluation, sampled pass@k, arbitrary-prompt generation and frozen dev loss evaluation ran on tiny checkpoints. Executable success was zero, as expected after three tiny steps.
- Small faithful/direct-patch vision comparison: two steps each, completed.
- ~25.7M digit pilot: one step, completed receipt; exit code 2 signals that its learning criterion was not met. Zero digit accuracy is retained in the receipt.
- Curriculum: three conditions × two steps; compositional diversity: two conditions × two steps. Both emitted receipts and were processed by the new descriptive structural analyzer.
- Attention microprofile: four implementations × two sequence lengths, forward/backward.
- Python compilation and Bash syntax checks completed.

## Repaired default size

- Total trainable parameter storage count: **12,625,740**.
- Architectural active-per-token estimate: **7,907,148**.

The 22M notebook is explicitly labeled a hypothetical arithmetic illustration. The larger vision pilot uses separate original-width dimensions, with 25,731,180 language parameters (including 12 added residual scalars).

## Evidence locations

`artifacts/cpu-smoke/` contains the training receipts, JSONL logs, configs, validation/evaluation receipts, profiler output and structural study outputs. Model/optimizer weights and environment folders are excluded from the ZIP. Run the commands yourself to create checkpoints.

Smoke receipts preserve source hashes and command arguments from the development moment when each run occurred; final source hashes are in `artifacts/SOURCE_MANIFEST.json`. Early smoke receipts may therefore have different source hashes from the final archive. The final pytest report tests the delivered code. Paths in captured receipts describe this verification environment; the README provides portable commands.

## What is still unmeasured

GPU throughput/VRAM, realistic training-to-quality targets, longer multi-seed learning curves, whether any individual change improves quality per second, and benefits of combining changes. No condition is selected as a proven winner. Automatic checkpoint resume and multi-GPU training are not implemented.

The code is prepared for local IDE review and controlled experiments; it is not a trained-model release or an established leaderboard record.
