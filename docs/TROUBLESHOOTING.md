# Experiment troubleshooting

## Colab plot subprocess exited with code 1 (reported 2026-10-19)

Observed command: `/content/glm-env/bin/python scripts/plot_benchmarks.py /content/modded-glm/runs/screen-20261019-203447 --output /content/modded-glm/runs/screen-20261019-203447/figures-204614`.

The supplied traceback contains `CalledProcessError` from the notebook's subprocess wrapper. It does **not** contain the plotting process's underlying traceback or stderr, so it does not establish a specific cause. It also does not mean training failed or its checkpoints were lost. The existing 22-condition CPU receipts render successfully locally; this does not reproduce or diagnose the reported GPU-session failure.

The updated notebook streams merged stdout/stderr, saves each command's full output under `runs/_command_logs/`, prints the log path on failure and preserves the original return code. Backup failures are reported without hiding the original command failure. Plot directories now use microsecond timestamps. The plotter reports and skips malformed, incomplete or non-pretraining receipt schemas; `skipped-receipts.json` records why. When nothing suitable exists, it explains how to plot `train.jsonl` instead. The main notebook displays every compatibility group, not just group-0.

### Recover without retraining

Keep the running runtime, CAMPAIGN directory, checkpoints and Drive backup. Update source code independently of training outputs:

- **Cloned repo:** from its root, use `git status --short`, then `git pull --ff-only` if your tracked source has no local edits. `runs/` is ignored and remains on disk. Retain custom source edits rather than discarding them.
- **Extracted project:** upload the current `modded-glm-flash-from-scratch.zip` and copy its updated `scripts/plot_benchmarks.py`, `scripts/plot_receipts.py`, `scripts/analyze_benchmarks.py` and `scripts/plot_loss.py` into the existing project. Do not replace/remove `runs/` or use a fresh runtime just to repair plots.

For an extracted project, this ordinary Colab cell updates only those files:

```python
from google.colab import files
from pathlib import Path
import zipfile
uploaded = files.upload()  # choose the updated project ZIP
name = next(n for n in uploaded if n.endswith('.zip'))
with zipfile.ZipFile(name) as archive:
    for relative in ['scripts/plot_benchmarks.py', 'scripts/plot_receipts.py',
                     'scripts/analyze_benchmarks.py', 'scripts/plot_loss.py']:
        (ROOT / relative).write_bytes(archive.read('modded-glm/' + relative))
uploaded.clear()
```

Then run this diagnostic/retry cell using the **existing** PY, ROOT and CAMPAIGN variables:

```python
import subprocess, uuid
from pathlib import Path

print('Campaign:', CAMPAIGN)
print('Receipts:', len(list(CAMPAIGN.rglob('training-receipt.json'))))
print('Training logs:', len(list(CAMPAIGN.rglob('train.jsonl'))))
subprocess.run([PY, '-c',
    'import sys,numpy,matplotlib; print(sys.executable); print(numpy.__version__,matplotlib.__version__)'], check=True)
PLOTS = CAMPAIGN / ('figures-retry-' + uuid.uuid4().hex[:10])
command = [PY, 'scripts/plot_benchmarks.py', str(CAMPAIGN), '--output', str(PLOTS)]
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
log = CAMPAIGN / ('plot-diagnostic-' + uuid.uuid4().hex[:10] + '.txt')
log.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
print('Diagnostic log:', log)
result.check_returncode()
from IPython.display import display, Image
for image in sorted(PLOTS.glob('group-*/*.png')):
    display(Image(filename=str(image)))
```

If the environment diagnostic reports missing plotting dependencies, install them into the same interpreter:

```python
subprocess.run([PY, '-m', 'pip', 'install', 'matplotlib>=3.8,<4'], check=True)
```

An unfinished run may have a usable `train.jsonl` before its final receipt exists:

```python
logs = sorted(CAMPAIGN.glob('*-seed*/train.jsonl'))
assert logs, 'No logged training steps in this campaign yet'
LOSS_PLOTS = CAMPAIGN / ('loss-retry-' + uuid.uuid4().hex[:10])
subprocess.run([PY, 'scripts/plot_loss.py', str(logs[0]), '--output', str(LOSS_PLOTS)], cwd=ROOT, check=True)
display(Image(filename=str(LOSS_PLOTS / 'loss.png')))
```

Other possible causes include an existing output folder, a wrong/empty campaign path, incompatible receipts, or dependency errors. The diagnostic log distinguishes them; none is confirmed solely by the reported wrapper traceback. Keep failed logs with the run evidence. Updating plot code after training does not change its stored source hashes or establish new training results.

## Custom objects in Colab

The notebook UI may use Python 3.13 while the model requires the isolated Python 3.12 environment. Ordinary cells manage campaigns/files. Start the project session and use `%%glm` for persistent model objects. Objects are not shared across interpreters; pass paths through `glm('', 'CHECKPOINT_PATH = '+repr(str(path)))` when needed. The session uses pipes rather than a TCP kernel and executes Python you provide; it is not a security sandbox.

A failed project cell displays its traceback and leaves the session alive. Interrupting a project cell stops the session to avoid a busy worker; rerun setup and object creation. `glm_session.close()` releases its objects. Avoid holding demonstration models on the GPU while timing a campaign.
