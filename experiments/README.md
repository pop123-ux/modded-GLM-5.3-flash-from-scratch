# Experiment menu

The main study is `run_ablation_matrix.py`. Default base: `configs/hybrid_baseline.json`. Use `configs/tiny.json` only for smoke checks.

| Arm | Change relative to base |
| --- | --- |
| `vuk` | Exact upstream model implementation with matched dimensions/data |
| `hybrid` | Repaired hybrid reference, alpha 1 and dropout 0 |
| `user_repaired` | Your alpha 0.01 plus dropout 0.1 combination |
| `small_alpha` | Alpha 0.01 only |
| `dropout` | Dropout 0.1 only |
| `streams1` / `streams4` | One / four streams |
| `standard_residual` | Standard block residual path and one stream |
| `sparse_sdpa` | Equivalent sparse mask via SDPA |
| `full_sdpa` | Full causal attention in every block |
| `qk_norm` | Head-wise query/key RMSNorm |
| `embedding_skip` | Zero-initialized learned embedding skips to every block |
| `softcap` | Tanh logit softcap at 15 |
| `balanced` | Differentiable balancing weight 0.01 |
| `router_z` | Router z-loss weight 0.001 |
| `muon` | Muon-style hidden-matrix updates + AdamW elsewhere |
| `cosine` | Cosine decay with five warmup steps |

```bash
python experiments/run_ablation_matrix.py --base-config configs/tiny.json --output runs/smoke --arms vuk,hybrid,balanced,sparse_sdpa,muon --seeds 42 --steps 3 --batch-size 2 --device cpu
python scripts/analyze_benchmarks.py runs/smoke --output runs/smoke/summary.json
```

Additional arms:

| Arm | Change relative to base |
| --- | --- |
| `real_tokens` | Padding-aware expert routing plus right-padding crop |
| `chunked_linear` | Chunked prefix state with backward recomputation |
| `dense_ffn` | Dense SwiGLU with active expert-width control |
| `bias_balanced` | Explicit expert selection-bias controller |
| `future_bytes` | Two additional future-byte training heads |

See [../research/EXTENSIONS.md](../research/EXTENSIONS.md) for evaluation criteria.

Other studies:

```bash
python experiments/profile_attention.py --output runs/attention.json --device cuda --lengths 64,128,256,512 --iterations 20
python experiments/pretraining_data_diversity.py --output runs/diversity --steps 200 --seeds 11,22,33
python experiments/pretraining_curriculum_order.py --output runs/curriculum --steps 200 --seeds 11,22,33
python scripts/train_vision.py --smoke-test --device cpu --output runs/vision.json
python experiments/full_25m_vision_digit_pilot.py --help
```

The curriculum/diversity and vision studies are adapted from upstream and now use the active model. Curriculum/diversity investigate generalization to withheld operation compositions; do not directly compare their losses with the 13-family coding benchmark. Attention profiling measures forward/backward cost only. Checkpoint transfer in the larger vision pilot does not itself prove a benefit over random initialization.

Interpret failed or noisy runs honestly. Add a bundled “best candidate” only after individual changes survive matched testing. No combination is designated a winner in this release.

Analyze actual structural receipts:

```bash
python experiments/analyze_structural.py runs/diversity/results.json runs/curriculum/results.json --output runs/structural-summary.json
```

The old analyzer filenames remain as entry points to this CLI. Original hardcoded upstream reports remain only in the exact baseline tree.
