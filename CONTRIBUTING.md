# Contributing experiments

The repository welcomes learning notes, correctness fixes, implementation improvements and training comparisons. A faster run becomes useful when someone else can reproduce its quality and timing.

## Start with a baseline

1. Run `python -m pytest -q`.
2. Reproduce the selected configuration using a fixed data seed, model seed, context and evaluation split.
3. Change one thing. Mark bundled changes clearly.
4. Save commands, config, environment, source/data hashes and all run receipts.
5. Compare time to the same next-byte validation target and report failed target runs too.

The project has no established global speed record, fixed GPU reference or universal target yet. Until those are agreed, present results as matched comparisons on named hardware. The FineWeb NanoGPT threshold does not apply to this byte-tokenized corpus.

## Reporting a result

Include the baseline and candidate receipts, hardware, Torch/Python versions, seeds, loss curves, wall-clock timing policy and peak memory when measured. Use at least three paired seeds for an exploratory training comparison, and more when the improvement is small or variance large.

Parameter count, active expert width, training-only heads and attention computation differ between arms. Keep those distinctions visible. Kernel microbenchmarks are useful but are not training records.

Use dev data for tuning. Evaluate final chosen configurations on confirm without selecting the winner from confirm. Template overlap remains in the synthetic coding tasks; do not describe results as general coding competence.

## Keeping the project learnable

Prefer readable changes with a clear rationale. Add meaningful correctness tests when changing mathematics, routing, labels, checkpoints or data flow. Preserve attribution and the pinned upstream baseline. Put speculative ideas in `research/IDEAS.md` and measured findings in records with their commands.

MIT contributions are welcome. See BENCHMARK_RULES.md and research/EXTENSIONS.md for the current protocol and options.

## Propose a research idea

Open a research-idea issue before implementing a substantial change. Describe the hypothesis, expected mechanism, comparison baseline and a result that would disprove it. Identify changed parameter/compute budgets and the minimum runnable experiment. A plausible idea can be useful even when it fails to improve training.

Use optional configuration flags for architectural experiments, preserve the existing defaults, and document an independent ablation before proposing a bundle. Include a CPU correctness check, then measure on named hardware. Put educational explanations in the theory index and experimental evidence in receipts. Credit Pop Alexandru's original notebook notes and attribute new contributions to their authors.
