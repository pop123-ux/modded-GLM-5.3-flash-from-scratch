# Command examples

Run commands from the project root; `python` should refer to your activated environment.

## Training and generation

```bash
python scripts/train_pretrain.py --config configs/user_repaired.json --output runs/my-design --steps 400 --device cuda
python scripts/generate.py --checkpoint runs/my-design/checkpoint-0400 --prompt "def example(x):" --max-new-tokens 64 --device cuda
python scripts/validate_loss.py --checkpoint runs/my-design/checkpoint-0400 --split confirm --per-family 8 --device cuda --output runs/my-design/confirm-loss.json
```

## Executable greedy and pass@k evaluation

```bash
python scripts/evaluate.py --checkpoint runs/my-design/checkpoint-0400 --split confirm --per-family 8 --max-new-tokens 192 --device cuda --output runs/my-design/confirm-code.json
python scripts/evaluate_passk.py --help
```

## Executable-reward RLOO

Start on a limited family set and inspect rewards. There is no update when every rollout receives the same reward.

```bash
python scripts/train_rl.py --initial-checkpoint runs/my-design/checkpoint-0400 --output runs/rloo --groups 32 --checkpoints 8,16,32 --group-size 4 --families increment,double,even --device cuda
```

## Optional diagnostics

```python
from dataclasses import replace
import torch
from glm53_flash import GLM53FlashFromScratch, ModelConfig

model = GLM53FlashFromScratch(replace(ModelConfig(), diagnostics=True))
logits, usage = model(torch.randint(4, 260, (1, 16)))
model.plot_stream_diagnostics(layer_idx=0)
model.plot_router(torch.randint(4, 260, (1, 16)), layer_idx=0)
```

Install `requirements-dev.txt` for plots. Keep diagnostics off in timed training. Model aliases `GLM53FlashFromScratch` and `ModdedGLM53FlashFromScratch` refer to the same active class.
