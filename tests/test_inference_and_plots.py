import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from glm53_flash import ModdedModel,ModelConfig,generate_text

class FixedModel(torch.nn.Module):
    def __init__(self,eos=False,bad=False):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()));self.config=SimpleNamespace(max_sequence_length=64);self.eos=eos;self.bad=bad
    def forward(self,x):
        logits=torch.zeros(*x.shape,260,device=x.device)
        logits[...,0]=100;logits[...,1]=99;logits[...,3]=98
        logits[...,69]=8;logits[...,70]=7
        if self.eos:logits[...,2]=20
        if self.bad:logits.fill_(float('nan'))
        return logits,torch.zeros(1)

def test_inference_masks_specials_and_restores_training_rng():
    model=FixedModel().train();state=torch.get_rng_state().clone()
    r=generate_text(model,'x',max_new_tokens=3)
    assert r.completion=='AAA' and r.text=='xAAA' and not r.stopped_on_eos
    assert model.training and torch.equal(state,torch.get_rng_state())
    assert generate_text(FixedModel(eos=True),'',max_new_tokens=3).token_ids==[2]

def test_sampling_filters_seed_and_errors():
    model=FixedModel()
    a=generate_text(model,'',max_new_tokens=12,temperature=1,top_k=2,seed=7)
    b=generate_text(model,'',max_new_tokens=12,temperature=1,top_k=2,seed=7)
    assert a.token_ids==b.token_ids and set(a.token_ids)<={69,70}
    assert generate_text(model,'',max_new_tokens=3,temperature=1,top_p=.01).token_ids==[69]*3
    assert generate_text(model,'',max_new_tokens=3,temperature=1,top_k=1).token_ids==[69]*3
    bad=FixedModel(bad=True).train()
    with pytest.raises(FloatingPointError):generate_text(bad,'',max_new_tokens=3)
    assert bad.training

@pytest.mark.parametrize('options',[dict(max_new_tokens=0),dict(temperature=-1),dict(temperature=float('nan')),dict(top_k=-1),dict(top_p=0),dict(top_p=1.1),dict(max_new_tokens=64)])
def test_generation_validation(options):
    with pytest.raises(ValueError):generate_text(FixedModel(),'',**options)

def test_custom_model_checkpoint_and_cli(tmp_path):
    cfg=ModelConfig(dim=16,layers=1,heads=2,expert_hidden=16,experts=2,top_k=1,streams=2,residual_dropout=0.)
    model=ModdedModel(cfg,qk_norm=True).eval();x=torch.tensor([[1,69,70]])
    model.save(tmp_path/'checkpoint',dict(purpose='test'))
    restored=ModdedModel.from_checkpoint(tmp_path/'checkpoint')
    assert restored.config.qk_norm and torch.equal(model(x)[0],restored(x)[0])
    prompt=tmp_path/'prompt.txt';prompt.write_text('def f(x):\n',encoding='utf-8')
    result=subprocess.run([sys.executable,'scripts/infer.py','--checkpoint',str(tmp_path/'checkpoint'),'--prompt-file',str(prompt),'--max-new-tokens','2','--json','--device','cpu'],check=True,capture_output=True,text=True)
    data=json.loads(result.stdout);assert data['prompt']==prompt.read_text() and len(data['token_ids'])<=2


def test_partial_loss_and_mixed_receipts(tmp_path):
    pytest.importorskip('matplotlib')
    from scripts.plot_loss import read_rows
    from scripts.plot_receipts import collect_receipts
    run=tmp_path/'hybrid-seed42';run.mkdir();log=run/'train.jsonl'
    log.write_text('{"step":1,"loss":5.0}\n{"step":2',encoding='utf-8')
    assert len(read_rows(log))==1
    source=Path('artifacts/research-smoke/final-extensions/hybrid-seed42/training-receipt.json')
    (run/'training-receipt.json').write_bytes(source.read_bytes())
    bad=tmp_path/'rl';bad.mkdir();(bad/'training-receipt.json').write_text(json.dumps(dict(status='complete',stage='executable_reward_rl')))
    corrupt=tmp_path/'broken';corrupt.mkdir();(corrupt/'training-receipt.json').write_text('{')
    valid,skipped=collect_receipts(tmp_path);assert len(valid)==1 and len(skipped)==2
    subprocess.run([sys.executable,'scripts/plot_loss.py',str(log),'--output',str(tmp_path/'single')],check=True,capture_output=True)
    subprocess.run([sys.executable,'scripts/plot_benchmarks.py',str(tmp_path),'--output',str(tmp_path/'ablations')],check=True,capture_output=True)
    assert (tmp_path/'single/loss.png').exists() and (tmp_path/'ablations/group-0/final-validation.svg').exists()
    assert len(json.loads((tmp_path/'ablations/skipped-receipts.json').read_text()))==2

def test_persistent_notebook_objects_and_error_recovery(capsys):
    from glm53_flash.notebook_bridge import NotebookSession
    with NotebookSession(sys.executable,Path.cwd()) as session:
        session.execute('from glm53_flash import ModdedModel,ModelConfig\nmodel=ModdedModel(dim=16,layers=1,heads=2,expert_hidden=16,experts=2,top_k=1,streams=2)\nvalue=40')
        session.execute("value+=2\nprint('VALUE',value)\nprint(type(model).__name__)")
        with pytest.raises(RuntimeError,match='Project cell failed'):session.execute("raise ValueError('expected failure')")
        session.execute('assert value==42\nassert model.config.dim==16')
    output=capsys.readouterr().out;assert 'VALUE 42' in output and 'ModdedModel' in output and 'expected failure' in output
