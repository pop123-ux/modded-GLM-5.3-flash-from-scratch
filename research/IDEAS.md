# Twenty ideas for a more interesting small-LLM lab

These are engineering/research priorities on a **1–10 scale**, not measured benefits or claims of algorithmic novelty. Scores combine architectural relevance, usefulness for learning, feasibility in this codebase, and how cleanly the idea can be tested. Established research ideas are adapted where appropriate.

The top five are implemented as **optional, independently selectable experiments** in v0.3.0. The remaining fifteen are proposals, not hidden or partial implementations.

| Rank | Idea | Score | Question / value | Status |
| --- | --- | --- | --- | --- |
| 1 | Route real tokens, not padding | 9.6/10 | How much expert compute and balancing noise comes from padding? Crop unused right-padding and exclude masked positions from dispatch/statistics. | Implemented: `real_tokens` |
| 2 | Chunk and checkpoint linear-attention state | 9.5/10 | Can the hybrid model trade recomputation for smaller saved activations while preserving its causal attention calculation? | Implemented: `chunked_linear` |
| 3 | A dense SwiGLU twin for the MoE | 9.3/10 | Does routing earn its overhead against a dense FFN with similar active expert matrix work? | Implemented: `dense_ffn` |
| 4 | Bias-based expert load control | 9.2/10 | Can explicit selection biases distribute tokens without adding a balancing-loss gradient? | Implemented: `bias_balanced` |
| 5 | Predict two or three bytes ahead | 9.0/10 | Does an auxiliary future-byte objective help the shared trunk learn useful code structure sooner? | Implemented: `future_bytes` |
| 6 | An expert specialization atlas | 8.9/10 | Map routing entropy and expert preferences by task family, prompt/completion region and token category. | Proposed |
| 7 | Mutation-based coding confirmation | 8.8/10 | Change constants, argument names and formatting to distinguish template memorization from transferable rules. | Proposed |
| 8 | A real-text track alongside synthetic code | 8.7/10 | Do results survive a fixed, openly redistributable text corpus and a richer task distribution? | Proposed |
| 9 | Learned shared-versus-routed expert gate | 8.6/10 | Let each token adjust reliance on shared computation versus specialist computation. | Proposed |
| 10 | Progressive context with a locked stream | 8.5/10 | Schedule context length while keeping the underlying byte stream fixed; measure time-to-quality. | Proposed |
| 11 | Expert-specific learning-rate diagnostics | 8.4/10 | Do infrequently selected experts need different update schedules or simply better routing? | Proposed |
| 12 | Training-tail weight averaging | 8.3/10 | Compare final weights with an explicitly timed EMA/tail average under the same validation protocol. | Proposed |
| 13 | Learned hybrid attention allocation | 8.2/10 | Which blocks actually need sparse/full attention versus prefix-state attention? | Proposed |
| 14 | Residual-stream diversity measurements | 8.1/10 | Track whether eight streams carry distinct information before adding any regularizer. | Proposed |
| 15 | Router temperature schedules | 8.0/10 | Explore sharp versus smooth mixture weights while monitoring dead experts and quality. | Proposed |
| 16 | Byte-category value embeddings | 7.9/10 | Give attention a small learned lexical signal for whitespace, digits and punctuation; test held-out text too. | Proposed |
| 17 | Cross-depth parameter sharing | 7.8/10 | Reuse blocks at different depths and measure storage, compute and quality separately. | Proposed |
| 18 | Confidence-based early exit | 7.7/10 | Can a small model skip later blocks on easy bytes without degrading the probability model? | Proposed |
| 19 | Self-distillation between residual streams | 7.6/10 | Test whether one stream can teach another while accounting for additional training cost. | Proposed |
| 20 | Multi-GPU expert-parallel track | 7.5/10 | When does communication become worthwhile compared with this single-device lab? High implementation cost at the current scale. | Proposed |

## Why these five first?

The first two address concrete inefficiencies visible in the implementation. The dense twin supplies a necessary control for the MoE story. Bias balancing and future-byte prediction introduce learning experiments with plausible mechanisms and clear failure modes. All five can be turned off and compared against the unchanged reference architecture.

This ranking does not declare a combined winner. In particular, extra future heads add training work; checkpointing recomputes activations; sparse token dispatch can incur indexing overhead. GPU and longer multi-seed measurements must decide whether they help.

See [EXTENSIONS.md](EXTENSIONS.md) for implementation details, commands and acceptance criteria.
