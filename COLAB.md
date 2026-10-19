# Google Colab: unified model experiment workspace

Use the cloned repo after publication or ZIP upload while private. All 22 model conditions, auxiliary studies and editable class cells live in this notebook. See [IDE_GUIDE.md](IDE_GUIDE.md) and [troubleshooting](docs/TROUBLESHOOTING.md).

# Modded-GLM: Colab experiments and honest figures
Run cells in order. Choose **Runtime → Change runtime type → GPU**, then connect. An NVIDIA T4 is suitable for initial small FP32 checks; resource availability varies. This notebook uses isolated Python 3.12 / PyTorch 2.5.1 rather than modifying Colab's notebook interpreter. No dataset download is needed: examples are generated locally.

Default source is ZIP upload while the repository is private. After publication, change SOURCE to `github`. Neither mode needs a GitHub token. Colab itself and GPU performance have not been executed/validated by the repository's CPU smoke tests.

Do not choose Run all blindly: the campaign and confirmation cells perform substantial work. Training checkpoints are saved, but the trainer does **not** resume optimizer training from them. Completed independent runs can be skipped; interrupted individual runs must restart under a new directory.

The experiment notebook needs the repository source in the runtime. Opening it from GitHub does not automatically clone the repo. Choose the cloned-repo path below after publication, or the separate upload fallback while private. Theory notebooks are primarily for reading; their vision snippets are illustrative, not standalone execution cells. For VS Code/Jupyter use IDE_GUIDE.md and notebooks/local_quickstart.ipynb.

**One experiment workspace:** all 22 registered model ablations plus optional auxiliary study families, checkpoint inference, loss/ablation figures and a persistent customizable model class are provided here. Auxiliary study switches prevent accidental long jobs; no separate research notebooks are needed.


## 1. Choose the project source

**Cloned repo:** after publication, set `SOURCE = 'github'`. This cell clones the repo into the Colab runtime, checks out REF, and sets ROOT to the cloned repo. All subsequent experiment commands use that checkout. Set REF to a commit SHA for an exact revision; `main` chooses the current branch head.

**Upload fallback:** while private, keep `SOURCE = 'upload'` and upload `modded-glm-flash-from-scratch.zip`. This creates an extracted project rather than a cloned repo. You do not upload a ZIP when using the clone option.


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

def run_script(script,*arguments,allowed_codes=(0,)):
    command=[PY,script,*map(str,arguments)]
    logs=ROOT/'runs'/'_command_logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/(Path(script).stem+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'.log')
    try:
        with log.open('w',encoding='utf-8') as handle:
            handle.write(json.dumps(command)+'\n');handle.flush()
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace',env=dict(os.environ,OMP_NUM_THREADS='1'))
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);handle.write(line);handle.flush()
                status=process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        print('Full command log:',log)
        if status not in allowed_codes:
            print(log.read_text(encoding='utf-8')[-12000:])
            raise subprocess.CalledProcessError(status,command,output=log.read_text(encoding='utf-8'))
        return status
    finally:
        try:backup()
        except Exception as exc:print('Backup failed; local results remain available:',exc)

```

## Interactive model class and customization
The training scripts use isolated Python 3.12. Colab's host kernel may instead use Python 3.13, which cannot import the pinned torch wheel. The following cell starts a persistent project Python session and registers **`%%glm`**. All `%%glm` cells share real Python objects across cells; ordinary cells continue to control files, campaigns and Drive. Do not expect a variable from one interpreter in the other. Rerunning session setup resets your interactive model objects; saved checkpoints remain on disk.

The notebook defines a subclass of the repository's full model rather than duplicating the architecture. You can change ModelConfig fields, access layers and parameters, override forward, add your own heads, write your own optimizer loop, or save/load checkpoints. Randomly initialized models are not useful generators until trained. Config/architecture changes require new compatible weights.


```python
# Use a persistent Python 3.12 session over pipes, without changing Colab's UI kernel.
import importlib.util
from IPython.core.magic import register_cell_magic
spec=importlib.util.spec_from_file_location('glm_bridge',ROOT/'glm53_flash/notebook_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
if 'glm_session' in globals():glm_session.close()
glm_session=bridge.NotebookSession(PY,ROOT)
@register_cell_magic
def glm(line,cell):
    return glm_session.execute(cell)
print('Use %%glm at the beginning of model cells. Objects persist until session close/restart.')

```

```python
%%glm
import torch
from dataclasses import replace
from glm53_flash import ModdedModel,ModelConfig,ByteTokenizer
class NotebookGLM(ModdedModel):
    """Your editable model class. Override forward or add modules here."""
    def __init__(self,config=None,**overrides):
        super().__init__(config,**overrides)
    def forward(self,input_ids,token_mask=None):
        return super().forward(input_ids,token_mask=token_mask)

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
config=ModelConfig(dim=32,layers=4,heads=4,expert_hidden=48,experts=4,top_k=2,
                   streams=4,residual_alpha=1.,residual_dropout=0.,max_sequence_length=512)
model=NotebookGLM(config,padding_aware_routing=True).to(DEVICE)
x=torch.tensor([ByteTokenizer().encode('def increment(x):',bos=True)],device=DEVICE)
logits,usage=model(x)
print(type(model).__name__,logits.shape,model.parameter_counts())

```

```python
%%glm
# A tiny next-byte optimizer example on one string, independent of timed campaigns.
import torch.nn.functional as F
tokens=torch.tensor([ByteTokenizer().encode('def increment(x):\n    return x + 1',bos=True,eos=True)],device=DEVICE)
x,labels=tokens[:,:-1],tokens[:,1:]
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
model.train();optimizer.zero_grad()
logits,usage=model(x)
language_loss=F.cross_entropy(logits.float().flatten(0,1),labels.flatten())
balance,z=model.router_losses()
future=model.future_prediction_loss(labels) if model.config.future_heads else language_loss.new_zeros(())
example_loss=language_loss+.01*balance+.001*z+.1*future
example_loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
optimizer.step();model.update_router_biases()
print('One next-byte demonstration update:',float(language_loss.detach()))
# This single example does not establish learning quality. The campaign below
# uses generated training data, held-out validation and reproducible receipts.
# Custom subclass source is not serialized by checkpoint helpers: version it.

```

Before timing campaigns, free demonstration weights and temporary tensors on the GPU. Use the class again later for trained-checkpoint inference. The class definition remains in the project Python session.


```python
%%glm
import gc
del model,optimizer,logits,usage,x,labels,tokens,language_loss,balance,z,future,example_loss
gc.collect()
if torch.cuda.is_available():torch.cuda.empty_cache()

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
print('Rough 22-arm × 400-step training-only budget:',r['training_seconds']*88,'seconds; evaluation/setup/backup add time')

```

## 5. All 22 model ablations in one campaign
The default campaign covers every registered model ablation, including the pinned Vuk baseline, all attention/residual/optimizer controls and the five research extensions. Set SELECTED_ARMS to a subset for a shorter run; do not create another notebook. Start with 400 steps and seed 42. All 22 arms cost substantially more than the earlier six-arm screen; use the pilot receipt to estimate your session budget. This is exploratory, not evidence from three seeds. Keep batch size, data seed, context, primary loss scope, validation split and eval frequency fixed. If the pilot runs out of memory, reduce batch size for **all** arms and start a new campaign.

The screen can use TARGET=None. After the pilot, you may set a reachable local dev-loss target, write down why you chose it, and freeze it before the paired runs. Do not copy NanoGPT's 3.28 threshold; this is a different task/tokenizer. Target hits are observed only at evaluation checkpoints and are not an official leaderboard target.

Each command backs up before the next run starts. Completed receipts are skipped when rerunning with the same CAMPAIGN. If an individual output exists without a complete receipt, inspect/retain its logs and use a new campaign/output name. Do not silently drop failures.


```python
sys.path.insert(0,str(ROOT))
from experiments.run_ablation_matrix import ARMS as ARM_DEFINITIONS
SELECTED_ARMS = list(ARM_DEFINITIONS)  # all 22; replace with a subset if needed
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
        if arm not in ARM_DEFINITIONS:raise ValueError(f'Unknown arm: {arm}')
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
# Microseconds avoid collisions when rerunning plotting cells. Never delete trained runs.
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
PLOTS = CAMPAIGN/f'figures-{STAMP}'
print('Completed receipts:',len(list(CAMPAIGN.rglob('training-receipt.json'))))
run_script('scripts/plot_benchmarks.py',CAMPAIGN,'--output',PLOTS)
run_script('scripts/analyze_benchmarks.py',CAMPAIGN,'--output',CAMPAIGN/f'summary-{STAMP}.json')
from IPython.display import display,Image
for group in sorted(PLOTS.glob('group-*')):
    print('Compatibility group:',group.name)
    for path in sorted(group.glob('*.png')):display(Image(filename=str(path)))
# Primary training loss (also works on an unfinished run with train.jsonl):
first = next((p for p in CAMPAIGN.glob('*-seed*/train.jsonl')),None)
if first is not None:
    SINGLE_PLOTS=CAMPAIGN/f'loss-{STAMP}'
    run_script('scripts/plot_loss.py',first,'--output',SINGLE_PLOTS)
    display(Image(filename=str(SINGLE_PLOTS/'loss.png')))

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

## Checkpoint inference — CLI and an instantiated object
Use a checkpoint selected on dev. These examples use the pilot's final checkpoint for workflow checking only. Generation is full-context autoregressive decoding, without a KV cache; it stops at EOS or the requested cap. It does not execute generated code. Greedy is temperature=0; positive temperature enables top-k/top-p sampling.


```python
INFER_CHECKPOINT=PILOT/'checkpoint-0100'
run_script('scripts/infer.py','--checkpoint',INFER_CHECKPOINT,'--prompt','def increment(x):',
           '--max-new-tokens',64,'--temperature',0,'--device','cuda','--json')
# Transfer only a path to the persistent model kernel; model objects stay there.
glm('', 'CHECKPOINT_PATH = '+repr(str(INFER_CHECKPOINT)))

```

```python
%%glm
trained=NotebookGLM.from_checkpoint(CHECKPOINT_PATH,device=DEVICE)
result=trained.generate('def increment(x):',max_new_tokens=64,temperature=.7,top_k=40,top_p=.9,seed=42)
print(result.text)
print('EOS:',result.stopped_on_eos)
# Continue customizing/inspecting `trained` here. Release before other GPU studies:
del trained
import gc;gc.collect()
if torch.cuda.is_available():torch.cuda.empty_cache()

```

## Auxiliary studies — same notebook, separate measurements
All repository study families are launchable below: attention time/memory, chunked saved-state storage, structural data diversity, curriculum order, RLOO, greedy/pass@k evaluation, miniature vision and the larger digit pilot. They use distinct tasks/receipts and are not merged into the coding-CE leaderboard. The structural studies currently support CPU/MPS rather than CUDA, so this notebook runs them on CPU. The vision comparison and larger pilot are separate model configurations.

Set the switches to True to execute the desired studies; leave costly studies off during an ablation session. No extra experiment notebook is required. Defaults are functional budgets, not converged results. The larger digit pilot may return exit code 2 when its learning criterion is not met; this is preserved as an unsuccessful learning result, not a software success claim.


```python
RUN_STUDIES=dict(attention=False,linear_storage=False,diversity=False,curriculum=False,
                 rl=False,code_evaluation=False,vision=False,large_vision_pilot=False)
STUDY_STEPS=20
STUDY_SEEDS=[42,43,44]
STUDIES=ROOT/'runs'/f'studies-{SESSION}'
STUDIES.mkdir(parents=True,exist_ok=True)
if RUN_STUDIES['attention']:
    run_script('experiments/profile_attention.py','--output',STUDIES/'attention.json',
               '--lengths','64,128,256','--iterations',10,'--batch-size',2,'--device','cuda')
if RUN_STUDIES['linear_storage']:
    run_script('experiments/profile_linear_memory.py','--output',STUDIES/'linear-storage.json',
               '--length',256,'--head-dim',16,'--chunk-size',32,'--device','cuda')
if RUN_STUDIES['diversity']:
    run_script('experiments/pretraining_data_diversity.py','--output',STUDIES/'diversity',
               '--steps',STUDY_STEPS,'--batch-size',2,'--sequence-length',320,
               '--seeds',','.join(map(str,STUDY_SEEDS)),'--device','cpu')
if RUN_STUDIES['curriculum']:
    run_script('experiments/pretraining_curriculum_order.py','--output',STUDIES/'curriculum',
               '--steps',STUDY_STEPS,'--batch-size',2,'--sequence-length',320,
               '--seeds',','.join(map(str,STUDY_SEEDS)),'--device','cpu')
structural=[p for p in [STUDIES/'diversity/results.json',STUDIES/'curriculum/results.json'] if p.exists()]
if structural:
    run_script('experiments/analyze_structural.py',*structural,'--output',STUDIES/'structural-summary.json')
if RUN_STUDIES['rl']:
    run_script('scripts/train_rl.py','--initial-checkpoint',INFER_CHECKPOINT,'--output',STUDIES/'rl',
               '--groups',8,'--checkpoints','8','--group-size',4,'--max-new-tokens',192,'--device','cuda')
if RUN_STUDIES['code_evaluation']:
    run_script('scripts/evaluate.py','--checkpoint',INFER_CHECKPOINT,'--split','dev',
               '--output',STUDIES/'greedy.json','--per-family',4,'--max-new-tokens',192,'--device','cuda')
    run_script('scripts/evaluate_passk.py','--checkpoint',INFER_CHECKPOINT,'--split','dev',
               '--output',STUDIES/'passk.json','--per-family',4,'--samples',8,'--max-new-tokens',192,'--device','cuda')
if RUN_STUDIES['vision']:
    run_script('scripts/train_vision.py','--architecture','both','--output',STUDIES/'vision',
               '--steps',STUDY_STEPS,'--batch-size',4,'--eval-examples',20,'--device','cuda')
if RUN_STUDIES['large_vision_pilot']:
    code=run_script('experiments/full_25m_vision_digit_pilot.py','--output',STUDIES/'large-vision',
                    '--steps',STUDY_STEPS,'--batch-size',2,'--eval-examples',20,'--device','cuda',allowed_codes=(0,2))
    print('Digit pilot exit code:',code)

```

## Optional RL variant screen and paired group-size confirmation
These cells replace the old machine-specific shell launchers for Colab. All eight screening conditions have the same 128-rollout budget. Confirmation compares group sizes 8 and 4 across three training seeds. Enable these explicitly only after selecting a pretrained checkpoint on dev; confirmation opens the confirm split and must not be used for tuning. These are expensive optional experiments, not extra notebooks. The script-specific statistical reports require their expected directory layout, supplied below.


```python
RUN_RL_VARIANTS=False
RUN_RL_GROUP_CONFIRMATION=False
RL_FAMILIES='increment,double,square,even,reverse'
RL_SEEDS=[31415,27182,16180]
RL_SCREEN=STUDIES/'rl-screen'
RL_CONFIRM=STUDIES/'rl-confirmation'

def rl_train(dest,groups,size,seed,reward='binary',penalty=-.1,temperature=.35):
    run_script('scripts/train_rl.py','--initial-checkpoint',INFER_CHECKPOINT,'--output',dest,
               '--groups',groups,'--checkpoints',groups,'--group-size',size,'--max-new-tokens',192,
               '--temperature',temperature,'--learning-rate',5e-5,'--train-scope','last-block-head',
               '--reward-mode',reward,'--invalid-penalty',penalty,'--families',RL_FAMILIES,
               '--tasks-per-family',8,'--seed',seed,'--device','cuda')
    return dest/f'checkpoint-{groups:04d}'
def rl_eval(checkpoint,output,split):
    run_script('scripts/evaluate.py','--checkpoint',checkpoint,'--split',split,'--per-family',8,
               '--max-new-tokens',192,'--families',RL_FAMILIES,'--seed',20260827,'--device','cuda','--output',output)
if RUN_RL_VARIANTS:
    rl_eval(INFER_CHECKPOINT,RL_SCREEN/'evaluations/pretrained-dev.json','dev')
    variants=[('baseline',16,8,'binary',-.1,.35),('partial',16,8,'case-fraction',-.1,.35),
              ('invalid-zero',16,8,'binary',0.,.35),('invalid-hard',16,8,'binary',-1.,.35),
              ('temp-low',16,8,'binary',-.1,.2),('temp-high',16,8,'binary',-.1,.8),
              ('group-4',32,4,'binary',-.1,.35),('group-16',8,16,'binary',-.1,.35)]
    for name,groups,size,reward,penalty,temp in variants:
        checkpoint=rl_train(RL_SCREEN/name,groups,size,31415,reward,penalty,temp)
        rl_eval(checkpoint,RL_SCREEN/f'evaluations/{name}-dev.json','dev')
    run_script('scripts/analyze_rl_variants.py','--screen-dir',RL_SCREEN,'--output',RL_SCREEN/'SCREENING-REPORT.md')
if RUN_RL_GROUP_CONFIRMATION:
    rl_eval(INFER_CHECKPOINT,RL_CONFIRM/'evaluations/pretrained-confirm.json','confirm')
    for seed in RL_SEEDS:
        for name,groups,size in [('baseline',16,8),('group-4',32,4)]:
            checkpoint=rl_train(RL_CONFIRM/f'{name}-seed-{seed}',groups,size,seed)
            rl_eval(checkpoint,RL_CONFIRM/f'evaluations/{name}-seed-{seed}-confirm.json','confirm')
    run_script('scripts/analyze_rl_group_confirmation.py','--run-dir',RL_CONFIRM,'--output',RL_CONFIRM/'CONFIRMATION-REPORT.md')

```

## Plots for auxiliary studies
The auxiliary plotter understands their separate receipt schemas. Run this after enabled studies finish; missing studies are listed as skipped. These charts are descriptive views, not directly comparable loss targets across tasks.


```python
if any(RUN_STUDIES.values()) or RUN_RL_VARIANTS or RUN_RL_GROUP_CONFIRMATION:
    AUX_PLOTS=STUDIES/('figures-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S-%f'))
    run_script('scripts/plot_studies.py',STUDIES,'--output',AUX_PLOTS)
    for path in sorted(AUX_PLOTS.glob('*.png')):display(Image(filename=str(path)))

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


## Close the interactive project session
After using your custom model objects, run `glm_session.close()` in an ordinary cell to release its memory. Interrupting a project cell stops the session; recreate it and its model objects to continue. `%%glm` accepts Python, not nested IPython magics; save figures to disk and display them from ordinary cells.

