# Five implemented research extensions

All new model flags default off. Existing configs/checkpoints remain loadable, and the original `vuk` implementation stays untouched. The matrix now has 22 arms. Primary validation remains next-byte cross entropy; auxiliary objectives do not replace it.

## 1. Real-token expert routing

- Config: `padding_aware_routing=true`.
- CLI: `--trim-padding` crops each batch to its final supervised target.
- Arm: `real_tokens`; preset: `configs/real_tokens.json`.

MoE dispatch gathers non-padding inputs, runs shared/routed experts on them, then scatters their outputs back. Usage, average router probabilities and z-loss exclude masked inputs. An all-masked batch produces zero router losses without NaNs. The trainer logs allocated input positions, non-padding positions and effective sequence length.

Cropping retains every supervised byte and EOS target. Tests compare token counts and validation loss before/after cropping. Attention is still computed on the rectangular cropped batch, and shorter rows still have right-padding. The mask governs experts, not arbitrary document boundaries or left-padding attention. When a custom embedding path is used, supply its explicit token mask if necessary.

Routing considers non-padding input bytes; this may include a short row's EOS input even where no subsequent target exists. Cropping removes a final EOS input if it has no supervised successor. With nonzero router objectives this can slightly alter routing statistics; the combined arm therefore is not advertised as a pure systems-only change under every objective.

**Measure:** next-byte quality, time-to-quality, processed versus non-padding positions, and expert load. Fewer padded positions do not automatically imply a faster GPU step.

## 2. Chunked, checkpointed linear attention

- Config: `linear_chunk_size=32`, `linear_checkpoint_chunks=true`.
- Arm: `chunked_linear`; preset: `configs/chunked_linear.json`.

Prefix key sums and key/value outer-product sums are computed inside each chunk. Their final states carry forward with gradients intact. Compact boundary states are cloned so views cannot retain the full chunk storage. Non-reentrant PyTorch checkpointing recomputes intermediates during backward.

Tests compare outputs, input gradients and layer parameter gradients against the original prefix implementation, including uneven final chunks. In exact arithmetic the attention calculation is the same; floating accumulation order can introduce small differences.

This is not constant-memory training: saved boundary states grow with chunk count, and outputs/projections still scale with sequence length. Recomputing can slow training. The included profiler measures unique storage referenced by tensors saved for backward in the isolated attention core, not total model memory or allocator peak.

```bash
python experiments/profile_linear_memory.py --output runs/linear-memory.json --length 256 --head-dim 16 --chunk-size 32
```

**Measure:** full-model GPU peak memory, time-to-quality and whether smaller activation storage enables larger useful batches.

## 3. Dense SwiGLU comparison

- Config: `ffn_mode="dense"`; optional explicit `dense_hidden`.
- Arm: `dense_ffn`; preset: `configs/dense_control.json`.

By default, dense hidden width is `(top_k + 1) * expert_hidden`, matching the combined hidden widths of selected specialists plus the shared expert. It approximately matches active expert projection matrix work; it does not match total stored capacity, routing work, nonlinear function class or all FLOPs.

Dense mode returns zero usage vectors because it has no experts to report. Router auxiliary losses are zero. Receipts identify the FFN mode and parameter counts. Compare a family of width/depth budgets before making a broad MoE claim.

**Measure:** validation quality and training time at comparable active expert projection work, with total capacity reported separately.

## 4. Bias-based load balancing

- Config: `router_bias_rate=0.001`; bounded by `router_bias_limit`.
- Arm: `bias_balanced`; preset: `configs/bias_balanced.json`.

Expert selection uses logits plus a non-gradient bias. Contribution weights use the original selected logits, so the controller changes selection rather than adding a learned mixture-weight offset.

After each optimizer step the trainer explicitly adjusts biases against the latest assignment distribution. Overused experts decrease; underused experts increase. The adjustment is centered and bounded. It is consumed once per step and disabled in evaluation. Biases are persistent checkpoint buffers; inference uses the learned values but never updates them. RL currently keeps this controller fixed.

This is a simple educational adaptation inspired by [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664), not an exact paper reproduction. The independent arm leaves existing padding behavior unchanged; combine with real-token routing in a separate experiment if desired. Keep `--balance-weight 0` for a clean auxiliary-loss-free comparison.

**Measure:** assignment dispersion, expert collapse, quality and time against the balancing-loss arm. Bias control can oscillate; tune its rate using dev runs.

## 5. Auxiliary future-byte prediction

- Config: `future_heads=2`; CLI: `--future-weight 0.1`.
- Arm: `future_bytes`; preset: `configs/future_bytes.json`.

The main head predicts the next byte. Two optional small hidden projections predict bytes two and three positions ahead through the shared vocabulary matrix. Auxiliary labels shift the already next-byte-aligned labels by one and two positions; padding, missing suffix targets and masked labels remain ignored. Each head's loss is averaged over its valid targets, then the head losses are averaged.

The shared trunk and future projections receive gradients. Standard generation and primary validation continue to use only the next-byte head. Training-only parameter counts are reported separately from the architectural inference count. This does not implement speculative decoding or faster inference.

Inspired by [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737). The tied vocabulary projections here are a small-model adaptation, not a reproduction of the paper's exact architecture. Additional training time is counted.

**Measure:** next-byte validation and executable success per second and per supervised main-task byte. Auxiliary horizon targets must not be counted as additional unique training examples.

## Run individually before combining

```bash
python experiments/run_ablation_matrix.py --output runs/extensions --arms hybrid,real_tokens,chunked_linear,dense_ffn,bias_balanced,future_bytes --seeds 42,43,44 --device cuda
python scripts/analyze_benchmarks.py runs/extensions --output runs/extensions/summary.json
```

`configs/research_bundle.json` combines four compatible changes; dense FFN is intentionally excluded because it is the architectural control. This bundled configuration is unproven. Add `--trim-padding` when using it, and compare it only after checking the individual arms.

For a small bundle execution check, make a copy with tiny dimensions/layers or use the tiny presets described in the tutorial. The production-sized bundle is not needed to verify the APIs.

## Combined functional smoke

The four compatible MoE additions have a small preset for checking the training path:

```bash
python scripts/train_pretrain.py --config configs/tiny_research_bundle.json --output runs/research-smoke --steps 3 --batch-size 2 --trim-padding --device cpu
```

Use `configs/research_bundle.json` for the larger research configuration. Test each addition independently before interpreting a combined run. The dense control is intentionally separate because it replaces routing.
