# Inference and custom models

Inference needs trained checkpoint files: `config.json`, `metadata.json` and `model.pt`. The repository does not ship trained weights. A three-step smoke checkpoint verifies the API; it is not a capable coding model or instruction/chat model.

## Command line

After the README's quickstart training command:

```bash
python scripts/infer.py --checkpoint runs/quickstart/checkpoint-0010 --prompt "def increment(x):" --max-new-tokens 64 --device auto
python scripts/infer.py --checkpoint runs/quickstart/checkpoint-0010 --prompt "def increment(x):" --temperature 0.7 --top-k 40 --top-p 0.9 --seed 42 --json
python scripts/infer.py --checkpoint runs/quickstart/checkpoint-0010 --prompt-file prompt.txt --output runs/prediction.json
python scripts/infer.py --checkpoint runs/quickstart/checkpoint-0010 --interactive
```

Interactive prompts are independent; this does not maintain a chat history. Use `/quit` to exit. `scripts/generate.py` remains a compatibility entry point. JSON output contains generated token IDs, the completion and whether EOS ended the result. Generated code is printed, never executed by inference.

## Python object

```python
from glm53_flash import ModdedModel, ModelConfig, generate_text

model = ModdedModel.from_checkpoint('runs/quickstart/checkpoint-0010', device='cpu')
result = model.generate('def increment(x):', max_new_tokens=64, temperature=0)
print(result.text)
```

To construct your own randomly initialized model:

```python
model = ModdedModel(
    dim=64, layers=4, heads=4, expert_hidden=128,
    experts=4, top_k=2, streams=4,
    residual_alpha=1.0, residual_dropout=0.0,
    padding_aware_routing=True,
)
```

The facade subclasses `ModdedGLM53FlashFromScratch`; all model parameters, layers and ordinary torch APIs remain accessible. Configuration overrides are validated by ModelConfig. `.generate()`, `.save(path)` and `.from_checkpoint(path)` are convenience methods. The standalone `generate_text(model, prompt, ...)` also works with the base model or a loaded pinned Vuk baseline.

You may subclass ModdedModel to add a custom head or override forward, then use your own optimizer/training loop. The main Colab defines `NotebookGLM` as an editable example and keeps its instances alive across `%%glm` cells. Custom source code is not serialized by a weights-only checkpoint: version it and instantiate the same subclass when restoring. The CLI loads the registered repository architecture; arbitrary custom subclasses need their own loader.

Changing dimensions, layer count, head topology or modules creates a different architecture and needs compatible training/weights. Loading a checkpoint is strict; architecture overrides are not applied to existing trained shapes. Pinned Vuk checkpoints use the base loader rather than the ModdedModel facade.

## Decoding limits

Temperature zero means greedy decoding; positive temperature enables sampling, optionally restricted by top-k and nucleus top-p. A local CPU RNG controls sampling without reseeding training. Seed repeatability is not a promise of identical numerics across hardware. Generation temporarily enters evaluation mode and restores the model's prior mode, even on failure.

Context is in bytes: BOS + prompt + requested generation cap must fit. There is no silent truncation or sliding context. Decoding recomputes the full context at every step and has no KV cache or speculative decoder. This prioritizes readability over serving throughput. For UTF-8, partial multi-byte outputs can decode with replacement characters, and generated-token count counts bytes/special tokens rather than words.
