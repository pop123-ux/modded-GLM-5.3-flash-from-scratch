import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from glm53_flash import GLM53FlashFromScratch, ModelConfig, ByteTokenizer
from glm53_flash.model import RMSNorm, SparseAttention, LinearAttention, CausalSelfAttention, apply_rot, RotaryEmbedding
from glm53_flash.runtime import save_checkpoint, load_checkpoint
from glm53_flash.evaluator import evaluate_source, validate_source
from glm53_flash.tasks import frozen_tasks, FAMILIES
from glm53_flash.optim import build_optimizer
from glm53_flash.data import batch_from_tasks
from glm53_flash.baseline import build_baseline

@pytest.fixture
def cfg():
    return ModelConfig(dim=32,layers=4,heads=4,expert_hidden=48,experts=4,streams=4,residual_alpha=1.,residual_dropout=0.,max_sequence_length=512)

@pytest.mark.parametrize('kwargs',[{'dim':126,'heads':6},{'top_k':5,'experts':4},{'streams':0},{'residual_dropout':1.},{'attention_mode':'bad'},{'sparse_stride':0}])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):ModelConfig(**kwargs)

def test_rmsnorm_scale_invariance():
    x=torch.randn(2,9,16)*4; norm=RMSNorm(16)
    assert torch.allclose(norm(x).square().mean(-1),torch.ones(2,9),atol=1e-5)
    assert torch.allclose(norm(x),norm(3*x),atol=1e-5)

@pytest.mark.parametrize('changes',[{}, {'sparse_backend':'sdpa'},{'attention_mode':'sdpa'},{'qk_norm':True}, {'residual_mode':'standard','streams':1},{'embedding_skip':True,'logit_softcap':2.}])
def test_forward_backward_and_causality(cfg,changes):
    torch.manual_seed(9);model=GLM53FlashFromScratch(replace(cfg,**changes)).eval()
    x=torch.randint(4,260,(2,16));other=x.clone();other[:,9:]=torch.randint(4,260,(2,7))
    logits,usage=model(x);new,_=model(other)
    assert torch.allclose(logits[:,:9],new[:,:9],atol=1e-5)
    assert torch.allclose(usage.sum(-1),torch.ones(cfg.layers))
    loss=F.cross_entropy(logits.flatten(0,1),x.flatten());loss.backward()
    assert model.embedding.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if changes.get('logit_softcap'):assert logits.abs().max()<=2

def test_sparse_sdpa_matches_gather_output_and_gradients(cfg):
    a=SparseAttention(cfg);b=SparseAttention(replace(cfg,sparse_backend='sdpa'));b.load_state_dict(a.state_dict())
    x=torch.randn(2,23,32,requires_grad=True);z=x.detach().clone().requires_grad_()
    ya=a(x);yb=b(z);assert torch.allclose(ya,yb,atol=2e-6,rtol=1e-5)
    ya.square().sum().backward();yb.square().sum().backward()
    assert torch.allclose(x.grad,z.grad,atol=3e-6,rtol=1e-5)

def test_rotary_cache_matches_reference():
    rope=RotaryEmbedding();q=torch.randn(2,13,4,8);k=torch.randn_like(q)
    actual=rope(q,k);expected=apply_rot(q,k)
    assert all(torch.allclose(a,b,atol=1e-6) for a,b in zip(actual,expected))
    cache=rope._cache;rope(q,k);assert rope._cache is cache
    rope(q[:,:5],k[:,:5]);assert rope._cache is not cache

def test_router_balancing_has_gradient(cfg):
    model=GLM53FlashFromScratch(cfg)
    with torch.no_grad():
        for l in model.layers:l.block.moe.router.weight[0].fill_(0.8)
    model(torch.randint(4,260,(2,32)))
    balance,z=model.router_losses();assert balance.requires_grad and z.requires_grad
    (balance+0.01*z).backward()
    assert sum(float(l.block.moe.router.weight.grad.abs().sum()) for l in model.layers)>0

def test_diagnostics_are_opt_in_and_populated(cfg):
    quiet=GLM53FlashFromScratch(cfg);quiet(torch.ones(1,7,dtype=torch.long))
    assert quiet.layers[0].last_input_weights is None
    model=GLM53FlashFromScratch(replace(cfg,diagnostics=True));model(torch.ones(1,7,dtype=torch.long))
    assert model.layers[0].last_stream_norms.shape==(1,7,4)
    assert model.layers[0].last_stream_similarity.shape==(4,4)
    assert model.layers[0].block.moe.last_router_probs.shape==(7,4)

@pytest.mark.parametrize('implementation',['modded','vuk'])
def test_checkpoint_roundtrip(cfg,tmp_path,implementation):
    model=(build_baseline(cfg) if implementation=='vuk' else GLM53FlashFromScratch(cfg)).eval()
    x=torch.randint(4,260,(1,7));before=model(x)[0]
    save_checkpoint(model,tmp_path/'checkpoint',{'implementation':implementation})
    after=load_checkpoint(tmp_path/'checkpoint',torch.device('cpu')).eval()(x)[0]
    assert torch.equal(before,after)
    with pytest.raises(FileExistsError):save_checkpoint(model,tmp_path/'checkpoint',{})

@pytest.mark.parametrize('kind',['adamw','muon'])
def test_optimizer_step_and_state(cfg,kind):
    model=GLM53FlashFromScratch(cfg);opt=build_optimizer(model,kind)
    params=[p for g in opt.param_groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})==len(list(model.parameters()))
    x=torch.randint(4,260,(1,8));before=model.embedding.weight.detach().clone()
    model(x)[0].square().mean().backward();opt.step()
    assert not torch.equal(before,model.embedding.weight)
    new=build_optimizer(model,kind);new.load_state_dict(opt.state_dict())

def test_completion_mask_and_no_truncation():
    task=frozen_tasks('dev',per_family=1)[0]
    x,full,completion=batch_from_tasks([task],256,'cpu')
    n=len(ByteTokenizer().encode(task.prompt,bos=True))
    assert (completion[0,:n-1]==-100).all()
    assert (completion[0,n-1:]==full[0,n-1:]).all()
    with pytest.raises(ValueError):batch_from_tasks([task],16,'cpu')

@pytest.mark.parametrize('completion',["\n    import os\n    return 1\n", "\n    return __import__('os')\n", "\n    while True: pass\n", "\n    return x.__class__\n"])
def test_verifier_rejects_unsafe_syntax(completion):
    task=frozen_tasks('dev',per_family=1)[0]
    assert evaluate_source(task,task.prompt+completion).status=='invalid'

def test_verifier_has_wall_clock_limit():
    task=frozen_tasks('dev',per_family=1)[0]
    result=evaluate_source(task,task.prompt+'\n    return x + 1\n',timeout=0.00001)
    assert result.status=='timeout'

def test_unicode_tokenizer_and_specials():
    t=ByteTokenizer();text='România — λ🙂';assert t.decode(t.encode(text,bos=True,eos=True))==text
    assert t.encode_pair('a','b')==[1,101,3,102,2]

def test_all_nonempty_notebooks_are_valid_json():
    for p in (Path(__file__).resolve().parents[1]/'theory').glob('*.ipynb'):
        if p.stat().st_size:assert 'cells' in json.loads(p.read_text())

def test_hybrid_baseline_matches_pinned_vuk(cfg):
    upstream=build_baseline(cfg).eval()
    modded=GLM53FlashFromScratch(cfg).eval()
    missing,unexpected=modded.load_state_dict(upstream.state_dict(),strict=False)
    assert all(name.endswith('.alpha') for name in missing)
    assert not unexpected
    x=torch.randint(4,260,(2,19))
    a,ua=upstream(x);b,ub=modded(x)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    assert torch.equal(ua,ub)

def test_rl_policy_gradient_and_leave_one_out(cfg):
    from scripts.train_rl import completion_log_probabilities, leave_one_out
    model=GLM53FlashFromScratch(cfg)
    values=completion_log_probabilities(model,[1,4,5],[[6,7,2],[8,2]],torch.device('cpu'))
    assert values.shape==(2,) and values.requires_grad
    advantages=leave_one_out([1.,0.]);assert advantages==[1.,-1.]
    objective=-(values*torch.tensor(advantages)).mean();objective.backward()
    assert model.embedding.weight.grad.abs().sum()>0

def test_default_context_holds_every_reference():
    tasks=frozen_tasks('dev',per_family=4)
    x,y,_=batch_from_tasks(tasks,320,'cpu')
    assert x.shape==(52,320)
