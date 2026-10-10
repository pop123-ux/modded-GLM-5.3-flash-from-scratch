# Google Colab experiment guide

Upload `colab/modded_glm_experiments.ipynb` through Colab → File → Upload notebook. Follow its cells in order. The same cells appear below for copying into a new notebook.

# Modded-GLM: Colab experiments and honest figures
Run cells in order. Choose **Runtime → Change runtime type → GPU**, then connect. An NVIDIA T4 is suitable for initial small FP32 checks; resource availability varies. This notebook uses isolated Python 3.12 / PyTorch 2.5.1 rather than modifying Colab's notebook interpreter. No dataset download is needed: examples are generated locally.

Default source is ZIP upload while the repository is private. After publication, change SOURCE to `github`. Neither mode needs a GitHub token. Colab itself and GPU performance have not been executed/validated by the repository's CPU smoke tests.

Do not choose Run all blindly: the campaign and confirmation cells perform substantial work. Training checkpoints are saved, but the trainer does **not** resume optimizer training from them. Completed independent runs can be skipped; interrupted individual runs must restart under a new directory.


## 1. Load the source
Upload `modded-glm-repaired.zip` (the updated archive containing this notebook and the plotter). For a public checkout, set SOURCE='github'. Set REF to a commit SHA to repeat an exact revision; `main` selects the latest version at clone time.


```python
from pathlib import Path
import subprocess, sys, os, json, zipfile, shutil, datetime
SOURCE = 'upload'  # 'github' after the repository is public
REF = 'main'
ROOT = Path('/content/modded-glm')
assert not ROOT.exists(), 'Source already exists. Reuse it by skipping this cell, or start a fresh runtime.'
if SOURCE == 'github':
    subprocess.run(['git','clone','https://github.com/pop123-ux/modded-GLM-5.3-flash-from-scratch.git',str(ROOT)],check=True)
    subprocess.run(['git','checkout',REF],cwd=ROOT,check=True)
    print('Source commit:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
elif SOURCE == 'upload':
    from google.colab import files
    uploaded = files.upload()
    archives = [n for n in uploaded if n.endswith('.zip')]
    assert len(archives) == 1, 'Upload exactly one project ZIP.'
    with zipfile.ZipFile(archives[0]) as archive:
        for name in archive.namelist():
            p = Path(name)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='modded-glm', name
        archive.extractall('/content')
    uploaded.clear()
else:
    raise ValueError('SOURCE must be upload or github')
assert (ROOT/'scripts/plot_benchmarks.py').exists(), 'Use the updated archive/commit with the Colab kit.'
os.chdir(ROOT)
print('Working directory:',ROOT)
```

## 2. Install the isolated environment
This downloads a separate Python and CUDA-enabled PyTorch wheel. It can take several minutes. Do not install the project's pinned torch into Colab's main interpreter. All project commands below use PY explicitly; notebook UI/Drive helpers keep using Colab's own Python.


```python
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
UV = shutil.which('uv')
assert UV
ENV = Path('/content/glm-env')
if not ENV.exists():
    subprocess.run([UV,'python','install','3.12'],check=True)
    subprocess.run([UV,'venv',str(ENV),'--python','3.12','--seed'],check=True)
PY = str(ENV/'bin/python')
subprocess.run([PY,'-m','pip','install','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cu121'],check=True)
subprocess.run([PY,'-m','pip','install','-r','requirements-dev.txt'],cwd=ROOT,check=True)
subprocess.run([PY,'-m','pip','freeze'],stdout=(ROOT/'colab-environment.txt').open('w'),check=True)
subprocess.run([PY,'-c',"import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(), 'No usable CUDA GPU: change runtime to GPU'; print(torch.cuda.get_device_name(0)); print('BF16 supported:',torch.cuda.is_bf16_supported())"],check=True)
subprocess.run(['nvidia-smi'],check=True)
```

## 3. Correctness first; optional persistent backup
Tests run on CPU even in the GPU runtime. A green suite does not verify CUDA execution. Mounting Drive stores a copy after each command; actual training stays on `/content` to avoid Drive I/O in timing. You can turn backup off, but then download results before the runtime ends.


```python
subprocess.run([PY,'-m','pytest','-q'],cwd=ROOT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
USE_DRIVE = True
BACKUP = None
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    BACKUP = Path('/content/drive/MyDrive/modded-glm-experiments')
    BACKUP.mkdir(parents=True,exist_ok=True)

def backup():
    if BACKUP is not None:
        if (ROOT/'runs').exists():
            shutil.copytree(ROOT/'runs',BACKUP/'runs',dirs_exist_ok=True)
        shutil.copy2(ROOT/'colab-environment.txt',BACKUP/'colab-environment.txt')

def run_script(script,*arguments):
    try:
        subprocess.run([PY,script,*map(str,arguments)],cwd=ROOT,check=True,
                       env=dict(os.environ,OMP_NUM_THREADS='1'))
    finally:
        backup()
```

## 4. GPU smoke check, then a baseline pilot
Run the tiny combined preset for three steps. Then run the real 12-layer baseline for 100 steps, batch 4, FP32. This is a starting budget, not a hardware promise. No `--amp` is used, so paired runs share the same precision even on a T4. Before increasing batch size, check the receipt's peak allocated VRAM and leave room for evaluation. The larger prefix attention can be much more expensive than the tiny preset.


```python
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
SMOKE = ROOT/'runs'/f'gpu-smoke-{SESSION}'
PILOT = ROOT/'runs'/f'pilot-{SESSION}'
run_script('scripts/train_pretrain.py','--config','configs/tiny_research_bundle.json','--output',SMOKE,
           '--steps',3,'--batch-size',2,'--eval-per-family',1,'--trim-padding','--device','cuda')
run_script('scripts/train_pretrain.py','--config','configs/hybrid_baseline.json','--output',PILOT,
           '--steps',100,'--batch-size',4,'--sequence-length',320,'--eval-every',20,
           '--eval-per-family',4,'--checkpoints','20,40,60,80','--seed',42,'--data-seed',42,'--device','cuda')
r = json.loads((PILOT/'training-receipt.json').read_text())
print('Pilot final dev CE:',r['validation_curve'][-1]['val_loss'])
print('Peak allocated GiB:',r['peak_vram_gib'])
print('Pilot timed seconds:',r['training_seconds'])
print('Rough 6-arm × 400-step training-only budget:',r['training_seconds']*24,'seconds; evaluation/setup/backup add time')
```

## 5. A matched six-arm screen
Compare hybrid against each of the five additions individually. Start with 400 steps and seed 42. This is exploratory, not evidence from three seeds. Keep batch size, data seed, context, primary loss scope, validation split and eval frequency fixed. If the pilot runs out of memory, reduce batch size for **all** arms and start a new campaign.

The screen can use TARGET=None. After the pilot, you may set a reachable local dev-loss target, write down why you chose it, and freeze it before the paired runs. Do not copy NanoGPT's 3.28 threshold; this is a different task/tokenizer. Target hits are observed only at evaluation checkpoints and are not an official leaderboard target.

Each command backs up before the next run starts. Completed receipts are skipped when rerunning with the same CAMPAIGN. If an individual output exists without a complete receipt, inspect/retain its logs and use a new campaign/output name. Do not silently drop failures.


```python
sys.path.insert(0,str(ROOT))
from experiments.run_ablation_matrix import ARMS as ARM_DEFINITIONS
SELECTED_ARMS = ['hybrid','real_tokens','chunked_linear','dense_ffn','bias_balanced','future_bytes']
SEEDS = [42]
STEPS = 400
BATCH = 4
TARGET = None  # optional numeric value chosen and frozen from pilot/dev work
CAMPAIGN = ROOT/'runs'/f'screen-{SESSION}'
BASE = json.loads((ROOT/'configs/hybrid_baseline.json').read_text())

def campaign(output,arms,seeds,steps,batch,target=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=dict(arms=arms,seeds=seeds,steps=steps,batch=batch,target=target,
                  data_seed=42,context=320,eval_every=20,eval_per_family=4,amp=False,base=BASE)
    spec=output/'campaign.json'
    if spec.exists():
        assert json.loads(spec.read_text())==settings, 'Changed settings: use a new campaign directory.'
    else:
        spec.write_text(json.dumps(settings,indent=2)+'\n')
    for arm in arms:
        changes,extra=ARM_DEFINITIONS[arm]
        config=output/f'{arm}.json'
        config.write_text(json.dumps(dict(BASE,**changes),indent=2)+'\n')
        for seed in seeds:
            dest=output/f'{arm}-seed{seed}'
            receipt=dest/'training-receipt.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('status')=='complete':
                print('Already complete:',dest.name);continue
            assert not dest.exists(), f'Incomplete run at {dest}; retain it and restart under a new output name.'
            arguments=['--config',config,'--output',dest,'--steps',steps,'--batch-size',batch,
                       '--seed',seed,'--data-seed',42,'--sequence-length',320,
                       '--eval-every',20,'--eval-per-family',4,'--device','cuda',*extra]
            if target is not None:arguments+=['--target-loss',target]
            run_script('scripts/train_pretrain.py',*arguments)
            print('Completed:',dest.name)
    return output

campaign(CAMPAIGN,SELECTED_ARMS,SEEDS,STEPS,BATCH,TARGET)
```

## 6. Plot measured data
Each curve is one seed. Bar charts show means, individual runs and sample SD only when at least two seeds exist. GPU memory is allocated CUDA memory, not total VRAM usage. Expert heatmaps summarize the last training batch, not corpus-wide specialization. Do not publish the tiny smoke figures as convergence evidence.

The plotter separates incompatible settings into groups. Exact image dimensions, chart labels and data come from Matplotlib, never from an image generator.


```python
PLOTS = CAMPAIGN/f'figures-{datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")}'
run_script('scripts/plot_benchmarks.py',CAMPAIGN,'--output',PLOTS)
SUMMARY = CAMPAIGN/f'summary-{datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")}.json'
run_script('scripts/analyze_benchmarks.py',CAMPAIGN,'--output',SUMMARY)
from IPython.display import display,Image
for name in ['validation-vs-step','validation-vs-time','final-validation','throughput','peak-memory','expert-load','time-to-target']:
    path=PLOTS/'group-0'/f'{name}.png'
    if path.exists():display(Image(filename=str(path)))
```

## 7. Confirmation plan (edit before running)
Choose the most promising arm from dev exploration, then compare it against hybrid using paired model seeds 42, 43, 44 and a fresh longer budget, initially 1,200 steps. Keep the selected target and settings fixed. These seeds match both arms, although different architectures can consume random numbers differently. Multiple seeds support a claim only if results are consistent; sample SD is not a confidence interval.

Do not tune with confirm results. If neither arm reaches the target, report both as not reached; do not substitute their training budgets for successful times. A Colab GPU/model change or runtime contention can invalidate a timing comparison even when names match; rerun matched conditions in one session and retain environment metadata.


```python
WINNER = 'real_tokens'  # EDIT based on dev results; this default is not a recommendation/winner
CONFIRM_RUNS = ROOT/'runs'/f'paired-{SESSION}'
# Uncomment only after choosing and freezing the experiment:
# campaign(CONFIRM_RUNS,['hybrid',WINNER],[42,43,44],1200,BATCH,TARGET)
# run_script('scripts/plot_benchmarks.py',CONFIRM_RUNS,'--output',CONFIRM_RUNS/'figures')
```

## 8. Frozen evaluation and executable checks
For each selected model/seed, choose a saved checkpoint using dev loss only. Matrix runs save initial/final weights; this pilot also saves intermediate checkpoints. Run the confirm split once after selection. Keep confirm losses and executable success separate. Generation starts at 48 bytes by default; the explicit allowance below is 192, chosen to fit prompt + generation within the 512-byte model context. If any prompt exceeds that combined budget, lower the allowance or evaluate a documented subset consistently for both arms.


```python
CHECKPOINT = PILOT/'checkpoint-0100'  # replace with selected saved checkpoint
FINAL_RESULTS = CHECKPOINT.parent/'selected-confirm'
# Uncomment only once the checkpoint/configuration is selected using dev:
# run_script('scripts/validate_loss.py','--checkpoint',CHECKPOINT,'--output',str(FINAL_RESULTS)+'-loss.json',
#            '--split','confirm','--per-family',8,'--batch-size',BATCH,'--sequence-length',320,'--device','cuda')
# run_script('scripts/evaluate.py','--checkpoint',CHECKPOINT,'--output',str(FINAL_RESULTS)+'-code.json',
#            '--split','confirm','--per-family',8,'--max-new-tokens',192,'--device','cuda')
```

## 9. Download and publish selectively
Download the full results bundle (includes weights; it can be large), or use the Drive backup. Put selected PNG/SVG figures into `assets/results/` and include commands, GPU, commit/source hashes, precision, seeds, context and target in their captions. Keep generated illustrations under `assets/illustrations/`. See `README_VISUALS.md` for image prompts and README placement. Never upload model weights to GitHub's ordinary source tree.


```python
backup()
archive=shutil.make_archive('/content/modded-glm-colab-results','zip',ROOT,'runs')
from google.colab import files
files.download(archive)
```

## Troubleshooting
- **Private clone fails:** use ZIP upload; the public clone path becomes available after you publish the repo.
- **No CUDA:** change runtime hardware to GPU and rerun GPU detection. A paid plan still does not guarantee a particular GPU.
- **Python/torch install:** rerun the isolated environment cell; do not replace the notebook kernel's torch. The cu121 wheel needs a compatible NVIDIA driver; if detection fails after GPU selection, retain the diagnostic output.
- **OOM:** halve BATCH for every comparison arm and use a new campaign. Leave context at 320: examples exceeding context are rejected rather than silently truncated.
- **Existing output:** choose a new campaign name. No optimizer-resume command is implemented.
- **Restart/disconnect:** recreate source/environment, mount Drive and copy a saved campaign into local `runs/` before plotting/skipping completed runs. Interrupted commands may have checkpoints but no final receipt.
- **No plot:** a receipt is written only at successful training completion; incomplete runs remain logs, not fabricated data points.
- **No time-to-target plot:** supply a frozen numeric TARGET before training. Without a target, compare actual loss curves and elapsed training.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [uv Python installation](https://docs.astral.sh/uv/guides/install-python/), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

