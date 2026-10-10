# Benchmark rules

1. Freeze the data seed, selected families, task definitions, tokenizer, validation split/size, label masking and evaluation procedure before comparing runs. Record hashes.
2. Separate dataset/architecture experiments from systems-only comparisons. Full SDPA, QK norm, residual variants, balancing and Muon are experimental changes, not verified improvements.
3. Compare on the same GPU/CPU, device count, PyTorch version, batch size and context for the first study. This trainer is single-process/single-device; DDP is not implemented.
4. Tune on `dev`; reserve `final`/`confirm` for chosen configurations. Original template overlap remains even with disjoint names.
5. Choose the target after establishing a baseline. Never copy the NanoGPT 3.28 target onto this byte-tokenized synthetic corpus.
6. Record training time to the first measured target crossing, not time per step alone. Crossing is observed only at evaluation intervals; report those intervals. Runs that never hit the target remain failures, not omitted rows.
7. Training timing synchronizes CUDA/MPS and includes data construction, forward, backward, clipping and optimizer steps. It excludes validation, logging and checkpoint saves. No optimizer update is used for untimed warmup. Report end-to-end elapsed time as well.
8. Keep at least three paired model seeds for exploratory comparisons; add runs when variance or effect size demands it. Descriptive means/stddev are not a statistical significance test.
9. For a submission retain source hashes, commands, config, dataset/validation hashes, loss curves, hardware receipt and environment. Compare losses under the same mask/tokenizer.
10. Keep kernel microbenchmarks separate from time-to-quality results. The included CPU smoke artifacts verify paths only and establish no GPU record.

The matrix runner changes one factor relative to its base configuration, except explicitly labeled bundled arms (`user_repaired`, standard residuals and the original Vuk implementation). Matrix variants share the data seed. Vuk's pinned implementation is evaluated on this lab's expanded corpus with matched dimensions; its historical published results are not copied into the active leaderboard.

Future-byte targets are auxiliary losses, not extra unique main-task examples. Dense controls must report stored capacity and active expert projection widths separately. Cropping may be grouped with fixed-padding runs when examples, target masks and evaluation are identical, but report the cropping flag and any routing-statistic changes. Router selection biases are updated only after optimizer steps; validation must not adapt them.
