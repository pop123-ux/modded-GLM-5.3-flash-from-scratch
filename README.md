# Modded-GLM Flash From Scratch

**A small hybrid-attention Mixture-of-Experts model to learn from, experiment with, and speedrun.**

[Read the theory](theory/README.md) · [VS Code / Jupyter](IDE_GUIDE.md) · [Run in Colab](COLAB.md) · [Explore experiments](research/EXTENSIONS.md) · [Contribute an idea](CONTRIBUTING.md)


I'm **Pop Alexandru**, a teenage aspiring AI researcher with a passion for LLMs: understanding how they work, building them from scratch, and exploring what makes them learn more efficiently. This repository is my way of turning that curiosity into something other people can run, study, and improve.

Built on [Vuk Rosić's GLM-5.3-Flash from Scratch](https://github.com/vukrosic/glm-5.3-flash-from-scratch), this project takes inspiration from the open experimentation and training-speedrun culture of [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt).

**Status:** CPU-verified with 88 passing tests and three-step tiny-model smoke runs across 22 experiment conditions, plus a combined-feature smoke run. GPU benchmarking and longer convergence studies are open work.

## Choose your starting point

| Your goal | Start here |
| --- | --- |
| Understand the architecture without training | [My educational theory notebooks](theory/README.md) |
| Check the project on your own computer | [The first CPU run](#your-first-run) |
| Train and plot experiments on a GPU | [Google Colab notebook and guide](COLAB.md) |
| Test an architectural hypothesis | [Experiment menu](experiments/README.md) and [research extensions](research/EXTENSIONS.md) |
| Propose a new research direction | [Ranked ideas](research/IDEAS.md) and [contribution guide](CONTRIBUTING.md) |

## Theory notebooks — written by Pop Alexandru

I wrote the notebooks in [`theory/`](theory/README.md) for educational purposes, sharing the explanations and examples I use to understand LLMs. They are open for readers to study, learn from, and suggest improvements to under the project's MIT license.

**[Browse the lessons without running any code](theory/README.md).** Each available lesson has a GitHub-readable page and its original Jupyter notebook. The index clearly marks topics that are still planned placeholders.

## Why another from-scratch LLM repository?

Different architectures offer different things to learn—and different bottlenecks to optimize.

This model combines **routed and shared experts, hybrid causal attention, RMSNorm, SwiGLU, rotary embeddings, and learned residual-stream mixing**. Together, they provide a compact way to explore ideas used in contemporary language models, with explicit PyTorch operations and educational notes alongside the implementation.

The interesting comparison is how a hybrid MoE model learns and uses compute relative to other small language-model designs. Does sparse routing help at this scale? When does full optimized attention beat a linear/sparse hybrid? Do extra residual streams earn their cost? Which optimizer reaches the same quality sooner?

Those are questions to measure. The project is a GLM-inspired educational implementation, not a reproduction of a released frontier model. Its byte-tokenized synthetic coding task also differs from NanoGPT's FineWeb benchmark, so their loss targets and published training times are not directly comparable.

## Join the training speedrun

**Can you reach the same validation quality faster?**

You're invited to try architectural changes, optimizer ideas, better kernels, or simpler implementations—and share reproducible results. The aim is to develop a community benchmark around this architecture while keeping the repository useful for learning.

A shared quality target and reference hardware have not been established yet. Start by reproducing a baseline, choose a documented target for a matched comparison, and keep the data and evaluation fixed.

When sharing an experiment or opening a pull request, include:

- The idea you tested and the baseline you compared against.
- Commands, configuration, seeds, hardware and software versions.
- Validation curves and time to reach the chosen quality target.
- Run receipts, memory measurements where available, and unsuccessful runs too.

Use multiple paired seeds before calling a change an improvement. Kernel speed alone does not establish faster learning. See [BENCHMARK_RULES.md](BENCHMARK_RULES.md) for the comparison protocol and [experiments/README.md](experiments/README.md) for the experiment menu.

## Use your preferred environment

Read theory notebooks without cloning or running code. For experiments, use the cloned repository in Colab or a local checkout in VS Code/Jupyter. Opening a notebook from GitHub does not clone its source into Colab. The ZIP upload option is a separate fallback.

[IDE_GUIDE.md](IDE_GUIDE.md) explains interpreter/kernel selection and setup. [The local quickstart notebook](notebooks/local_quickstart.ipynb) runs a tiny CPU experiment without Colab-specific paths or imports.

## Setup

Use Python 3.10–3.12; Python 3.12 is recommended for the pinned PyTorch release.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

For CPU-only Linux/Windows, install the CPU wheel first:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Install `requirements-dev.txt` for optional plots. Render completed run receipts with `python scripts/plot_benchmarks.py runs/YOUR_CAMPAIGN --output runs/YOUR_CAMPAIGN/figures`.

## One notebook for model experiments

The main [Colab experiment notebook](colab/modded_glm_experiments.ipynb) is the single research workspace: **all 22 registered model ablations**, custom configuration, primary/auxiliary loss plots, comparison charts, inference, attention/storage profiles, structural data/curriculum studies, RLOO screens and paired group-size confirmation, and vision pilots. Auxiliary studies are selectable and keep their own measurements; you do not need a separate notebook for each experiment. The local quickstart remains a small IDE execution check.

It also defines an editable `NotebookGLM` class backed by the full repository model. Use the persistent `%%glm` project session to instantiate, inspect, subclass or train objects with your own code. This keeps model execution in Python 3.12 even when Colab’s UI uses another version.

## Run in Google Colab

[Open the experiment notebook](https://colab.research.google.com/github/pop123-ux/modded-GLM-5.3-flash-from-scratch/blob/main/colab/modded_glm_experiments.ipynb) once the repository is public. While private, download/upload the notebook and use its ZIP-source option.

For ZIP upload, use **`modded-glm-flash-from-scratch.zip`**.

[COLAB.md](COLAB.md) walks through the isolated Python/CUDA setup, GPU smoke check, baseline pilot, six-arm screen, paired confirmation, plotting and backups. [README_VISUALS.md](README_VISUALS.md) contains a visual plan and five image-generation prompts.

## Your first run

```bash
python scripts/train_pretrain.py --config configs/tiny.json --output runs/quickstart --steps 10 --batch-size 2 --eval-every 5 --device cpu
python scripts/validate_loss.py --checkpoint runs/quickstart/checkpoint-0010 --output runs/quickstart/dev-loss.json --split dev --per-family 1 --device cpu
python scripts/evaluate.py --checkpoint runs/quickstart/checkpoint-0010 --split dev --families increment,double --per-family 1 --output runs/quickstart/dev-code.json --device cpu
```

Ten tiny steps verify the workflow; they do not produce a capable coding model. Use `dev` while experimenting and reserve `final`/`confirm` for evaluating selected configurations.

Context is measured in **bytes**, not words. Training defaults to 320 input positions; oversized examples raise an error rather than being silently truncated.

## Inference

Use a saved checkpoint after training:

```bash
python scripts/infer.py --checkpoint runs/quickstart/checkpoint-0010 --prompt "def increment(x):" --max-new-tokens 64 --device auto
```

Greedy, temperature/top-k/top-p sampling, UTF-8 prompt files, interactive prompts and JSON results are supported. [INFERENCE.md](INFERENCE.md) documents decoding and checkpoint limits, plus the `ModdedModel` Python API for custom instances. No trained weights are bundled.

## Plot losses and compare experiments

```bash
# A single run, including partial training logs before a final receipt exists:
python scripts/plot_loss.py runs/quickstart --output runs/quickstart/loss-figures
# Completed, compatible coding-pretraining arms:
python scripts/plot_benchmarks.py runs/ablations --output runs/ablations/figures
# Distinct auxiliary study schemas (RL, vision, structural studies and profiles):
python scripts/plot_studies.py runs/studies --output runs/studies/figures
```

Install `requirements-dev.txt` first. Plots export PNG/SVG and source metadata. Use a new output directory when rerunning. Validation-versus-step/time, final loss, throughput, available GPU memory and expert loads are separated from single-run auxiliary losses. The main Colab launches these plotters and shows every compatibility group. [Troubleshooting](docs/TROUBLESHOOTING.md) documents the reported Colab plotting failure and how to retry without retraining.

## Model configurations

| Config | Purpose |
| --- | --- |
| `configs/user_repaired.json` | Eight-stream design with initial residual scale 0.01 and dropout 0.1 |
| `configs/hybrid_baseline.json` | Hybrid reference with initial residual scale 1 and dropout 0 |
| `configs/sdpa.json` | Full causal SDPA attention experiment |
| `configs/tiny.json` | Fast functional checks: 32-wide, four-layer model |

The default model has 128 dimensions, 12 layers, four heads, four routed experts, top-2 routing plus a shared expert, and eight residual streams: **12.63M total parameters**, with an architectural active-per-token estimate of **7.91M**.

`--implementation vuk` uses the pinned upstream model implementation with matched dimensions and the current corpus. This is a controlled comparison in this lab, not a replication of Vuk's published results.

## Research directions

The lab includes **22 selectable experiment conditions**. Padding-aware dispatch, chunked linear attention, a dense SwiGLU control, bias-based expert balancing and future-byte prediction provide five additional ways to study this architecture. Each is optional and has a documented comparison protocol.

Explore [the 20 ranked ideas](research/IDEAS.md), [the five implemented extensions](research/EXTENSIONS.md), and [CONTRIBUTING.md](CONTRIBUTING.md) to choose a question or share a reproducible run. The scores prioritize useful experiments; they do not promise better performance.

## Run experiments

```bash
# Inspect the planned commands first.
python experiments/run_ablation_matrix.py --output runs/ablation-plan --dry-run

# Run matched conditions with three model seeds.
python experiments/run_ablation_matrix.py --output runs/ablations --seeds 42,43,44 --steps 400 --device cuda
python scripts/analyze_benchmarks.py runs/ablations --output runs/ablations/summary.json
```

Select conditions with `--arms`. Available experiments cover attention and memory strategies, MoE/dense controls, stream counts, residual connections, routing losses and biases, QK normalization, embedding skips, logit softcaps, future-byte objectives, optimizers and learning-rate schedules. None is designated a proven winner.

The repository also includes executable-reward RLOO, greedy/pass@k evaluation, attention profiling, compositional data-diversity and curriculum studies, and a miniature vision path. See [EXAMPLES.md](EXAMPLES.md) for commands.

## Learn the architecture

- [TUTORIAL.md](TUTORIAL.md): a practical first experiment.
- [DOCUMENTATION.md](DOCUMENTATION.md): architecture, implementation and measurement details.
- [Theory notebooks](theory/README.md): original educational notes by Pop Alexandru, readable lesson exports, and clearly marked planned topics.
- `glm53_flash/`: model, tokenizer, data, optimizers, runtime, verifier and vision.
- `scripts/`, `experiments/`, `tests/`: training workflows, experiments and correctness checks.
- `artifacts/`: actual CPU smoke receipts and verification evidence; no trained weights.
- `baselines/upstream/`: pinned upstream source for attribution and reproducibility.

[REPORT.md](REPORT.md) records what has been verified and what remains unmeasured. A Medium article explaining the project and inviting more experiments is planned.

## Author and acknowledgments

Created and maintained by **Pop Alexandru**. Thank you to **Vuk Rosić** for the original educational repository, and to **Keller Jordan and the modded-nanogpt contributors** for the inspiration to make training optimization open, measurable, and collaborative.

MIT licensed. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for attribution.

---

- **Medium profile** — [medium.com/@Pop123](https://medium.com/@Pop123)
- **Hugging Face** — [pop123ux](https://huggingface.co/pop123ux)
