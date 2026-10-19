# Verification report — v0.4.0 — 2026-10-19

The project passes **88 CPU tests**. This release adds the shared inference API, customizable model facade and notebook session, single-run loss plots, auxiliary-study figures and diagnostics for the reported Colab plotting failure.

## Executed checks

- Full pytest suite: 88 passed. Covers the prior architecture/data/training checks plus EOS stopping, special-token masking, greedy and filtered sampling, sampling seed repeatability, preservation of training mode/global RNG, invalid contexts/options, model checkpoint round trips, prompt-file/JSON CLI inference, incomplete JSONL logs, mixed receipt schemas and persistent cross-cell model objects/error recovery.
- All 22 existing tiny-model CPU receipts rendered with the current ablation plotter and were processed by the analyzer. These are historical three-step runs, not new training or GPU speedup measurements.
- Auxiliary plotter rendered eight figures from existing attention, curriculum/diversity, RL, vision and executable-evaluation receipts. Separate task metrics remain separate.
- Main notebook: all 36 code/Markdown cells are valid notebook JSON, with Python code cells parsed (the `%%glm` cell marker is stripped for Python parsing).
- The actual notebook's editable model-class, cross-cell optimizer/gradient example and GPU-memory cleanup cells ran in a persistent Python 3.12 session on CPU. The session helper uses subprocess pipes, retains objects and surfaces exceptions; it does not require a TCP kernel or modify Colab's UI interpreter.
- The CLI and plot integration tests are enabled in CI by installing the optional plotting dependencies. The earlier v0.3 hosted Linux/Windows run passed; hosted results for this revision are reported by GitHub Actions.

Environment: Linux x86_64, Python 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.5.4, Matplotlib 3.11.2. CUDA, Colab, BF16, MPS and long convergence studies were not executed here. The main notebook includes launch paths for every registered ablation and auxiliary study, but this does not claim every long campaign has been run.

## Reported plot failure

The supplied 2026-10-19 Colab traceback contains `CalledProcessError` and exit code 1, without the child process's underlying exception. Its exact cause is unconfirmed. `docs/TROUBLESHOOTING.md` records the observed command and provides non-destructive source-update, diagnostic and plot-retry cells. The notebook now streams/logs stdout and stderr, uses unique output paths and lists skipped receipts. None of these changes requires retraining stored checkpoints.

## Interpretation

No trained weights are packaged and no new speed record, quality gain or architecture novelty is claimed. Random model objects and short smoke checkpoints are API demonstrations. Inference recomputes full context rather than using a KV cache. Keep dev selection separate from confirmation, preserve failed runs, and follow BENCHMARK_RULES.md for matched comparisons. Earlier verification history is retained in artifacts/PREVIOUS_REPORT_v0.3.md and artifacts/PREVIOUS_REPORT_v0.2.md.
