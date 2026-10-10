"""Tests for the five opt-in research additions, including numerical equivalence."""
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from glm53_flash import ModelConfig, GLM53FlashFromScratch
from glm53_flash.model import LinearAttention, SparseMixtureOfExperts, DenseFeedForward
from glm53_flash.linear import chunked_linear_attention
from glm53_flash.objectives import future_byte_loss
from glm53_flash.data import batch_from_tasks
from glm53_flash.tasks import frozen_tasks
from glm53_flash.runtime import save_checkpoint, load_checkpoint
from scripts.train_pretrain import validation

@pytest.fixture
def small():
    return ModelConfig(dim=32,layers=4,heads=4,expert_hidden=48,experts=4,streams=4,
                       residual_alpha=1.,residual_dropout=0.,max_sequence_length=512)

def reference(q,k,v):
    state=torch.einsum('bthd,bthe->bthde',k,v).cumsum(1)
    numerator=torch.einsum('bthd,bthde->bthe',q,state)
    denominator=torch.einsum('bthd,bthd->bth',q,k.cumsum(1))[...,None]
    return numerator/denominator.clamp_min(1e-6)

@pytest.mark.parametrize('chunk_size',[1,3,8,64])
@pytest.mark.parametrize('checkpoint_chunks',[False,True])
def test_chunked_values_and_all_input_gradients(chunk_size,checkpoint_chunks):
    torch.manual_seed(7)
    first=[(torch.rand(2,19,2,4,dtype=torch.float64)+0.1).requires_grad_() for _ in range(3)]
    second=[x.detach().clone().requires_grad_() for x in first]
    a=reference(*first);b=chunked_linear_attention(*second,chunk_size,checkpoint_chunks)
    assert torch.allclose(a,b,atol=1e-10,rtol=1e-10)
    a.square().sum().backward();b.square().sum().backward()
    for x,y in zip(first,second):assert torch.allclose(x.grad,y.grad,atol=1e-9,rtol=1e-9)

def test_checkpointed_chunks_save_fewer_tensor_elements():
    def count(fn):
        sizes=[]
        def pack(t):sizes.append(t.numel());return t
        q,k,v=[torch.rand(1,64,2,8,requires_grad=True) for _ in range(3)]
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            out=fn(q,k,v)
        out.sum().backward()
        return sum(sizes)
    full=count(reference)
    chunked=count(lambda q,k,v:chunked_linear_attention(q,k,v,8,True))
    assert chunked<full

@pytest.mark.parametrize('checkpoint_chunks',[False,True])
def test_linear_layer_parameter_gradient_equivalence(small,checkpoint_chunks):
    a=LinearAttention(small);b=LinearAttention(replace(small,linear_chunk_size=5,linear_checkpoint_chunks=checkpoint_chunks))
    b.load_state_dict(a.state_dict());x=torch.randn(2,17,32)
    aa=a(x);bb=b(x);assert torch.allclose(aa,bb,atol=2e-6)
    aa.square().sum().backward();bb.square().sum().backward()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=3e-6,rtol=1e-5)

def test_masked_moe_equals_only_real_tokens(small):
    moe=SparseMixtureOfExperts(replace(small,diagnostics=True))
    x=torch.randn(2,11,32,requires_grad=True);mask=torch.zeros(2,11,dtype=torch.bool);mask[0,:5]=True;mask[1,:8]=True
    out,usage=moe(x,token_mask=mask);balance=moe.aux_loss;z=moe.z_loss
    assert (out[~mask]==0).all()
    assert (moe.last_router_probs[~mask.flatten()]==0).all()
    compact,other=moe(x[mask][None])
    assert torch.allclose(out[mask],compact[0],atol=1e-6)
    assert torch.equal(usage,other)
    assert torch.allclose(balance,moe.aux_loss) and torch.allclose(z,moe.z_loss)
    out.sum().backward();assert (x.grad[~mask]==0).all()

def test_all_padding_router_is_finite(small):
    moe=SparseMixtureOfExperts(replace(small,router_bias_rate=0.1))
    x=torch.randn(1,4,32,requires_grad=True)
    out,usage=moe(x,token_mask=torch.zeros(1,4,dtype=torch.bool))
    assert out.eq(0).all() and usage.eq(0).all()
    assert torch.isfinite(moe.aux_loss) and torch.isfinite(moe.z_loss)
    (out.sum()+moe.aux_loss).backward();moe.update_bias();assert moe.router_bias.eq(0).all()

@pytest.mark.parametrize('scope',['full','completion'])
def test_cropping_preserves_targets_and_validation(small,scope):
    tasks=frozen_tasks('dev',per_family=1)[:3]
    full=batch_from_tasks(tasks,320,'cpu',scope)
    short=batch_from_tasks(tasks,320,'cpu',scope,trim_padding=True)
    width=short[0].shape[1];assert width<320
    for a,b in zip(full,short):assert torch.equal(a[:,:width],b)
    assert (full[1][:,width:]==-100).all()
    model=GLM53FlashFromScratch(replace(small,padding_aware_routing=True))
    a=validation(model,tasks,320,2,torch.device('cpu'),scope)
    b=validation(model,tasks,320,2,torch.device('cpu'),scope,trim_padding=True)
    assert a['val_tokens']==b['val_tokens']
    assert abs(a['val_loss']-b['val_loss'])<1e-6

def test_dense_control_matches_active_expert_width(small):
    ffn=DenseFeedForward(small)
    assert sum(p.numel() for p in ffn.parameters())==3*small.dim*small.expert_hidden*(small.top_k+1)
    model=GLM53FlashFromScratch(replace(small,ffn_mode='dense'))
    x=torch.randint(4,260,(2,13));logits,usage=model(x)
    assert usage.eq(0).all() # No fabricated expert distribution for the dense path.
    counts=model.parameter_counts();assert counts['total']==counts['active_per_token_estimate']
    loss=F.cross_entropy(logits.flatten(0,1),x.flatten());loss.backward()
    assert model.layers[0].block.moe.dense.up.weight.grad.abs().sum()>0


def test_bias_controller_is_explicit_centered_and_once_per_step(small):
    moe=SparseMixtureOfExperts(replace(small,router_bias_rate=0.1))
    moe(torch.randn(2,9,32));assert moe.router_bias.eq(0).all()
    moe.pending_usage=torch.tensor([1.,0.,0.,0.]);moe._last_valid_tokens=18
    moe.update_bias();bias=moe.router_bias.clone()
    assert bias[0]<0 and (bias[1:]>0).all() and abs(float(bias.sum()))<1e-6
    moe.update_bias();assert torch.equal(bias,moe.router_bias)
    moe.eval();moe.pending_usage=torch.tensor([1.,0.,0.,0.]);moe.update_bias()
    assert torch.equal(bias,moe.router_bias)


def test_router_bias_only_changes_selection(small):
    moe=SparseMixtureOfExperts(replace(small,top_k=1,router_bias_rate=0.1))
    with torch.no_grad():moe.router.weight.zero_();moe.router_bias[2]=1.
    x=torch.randn(1,3,32);out,usage=moe(x)
    assert usage.tolist()==[0.,0.,1.,0.]
    assert torch.allclose(out,moe.shared(x)+moe.experts[2](x),atol=1e-6)


def test_future_byte_offsets_masking_and_gradients(small):
    labels=torch.tensor([[1,2,3,-100,-100]])
    heads=[torch.zeros(1,5,8,requires_grad=True) for _ in range(2)]
    with torch.no_grad():heads[0][0,0,2]=20;heads[0][0,1,3]=20;heads[1][0,0,3]=20
    loss=future_byte_loss(heads,labels,heads[0]);assert float(loss)<1e-6
    loss.backward();assert heads[0].grad[:,2:].eq(0).all() and heads[1].grad[:,1:].eq(0).all()
    model=GLM53FlashFromScratch(replace(small,future_heads=2));x=torch.randint(4,260,(2,13));model(x)
    model.future_prediction_loss(x).backward()
    assert all(p.weight.grad.abs().sum()>0 for p in model.future_predictors)
    assert model.embedding.weight.grad.abs().sum()>0
    counts=model.parameter_counts();assert counts['training_only_parameters']==2*32*32


def test_future_loss_all_ignored_does_not_nan():
    anchor=torch.randn(3,requires_grad=True);heads=[torch.randn(1,3,8,requires_grad=True)]
    loss=future_byte_loss(heads,torch.full((1,3),-100),anchor)
    assert loss==0 and loss.requires_grad;loss.backward()

@pytest.mark.parametrize('changes',[{'padding_aware_routing':True}, {'linear_chunk_size':3,'linear_checkpoint_chunks':True}, {'ffn_mode':'dense'}, {'router_bias_rate':0.1}, {'future_heads':2}])
def test_extension_causality_checkpoint_roundtrip(small,changes,tmp_path):
    model=GLM53FlashFromScratch(replace(small,**changes)).eval()
    x=torch.randint(4,260,(1,14));other=x.clone();other[:,8:]=torch.randint(4,260,(1,6))
    logits,_=model(x);future=[v.detach().clone() for v in model.future_logits()]
    changed,_=model(other);assert torch.allclose(logits[:,:8],changed[:,:8],atol=1e-5)
    for a,b in zip(future,model.future_logits()):assert torch.allclose(a[:,:8],b[:,:8],atol=1e-5)
    save_checkpoint(model,tmp_path/'checkpoint',{});restored=load_checkpoint(tmp_path/'checkpoint',torch.device('cpu')).eval()
    assert torch.equal(model(x)[0],restored(x)[0])

@pytest.mark.parametrize('changes',[{'linear_checkpoint_chunks':True},{'ffn_mode':'bad'},{'router_bias_rate':-1},{'ffn_mode':'dense','router_bias_rate':0.1},{'future_heads':-1}])
def test_extension_config_validation(changes):
    with pytest.raises(ValueError):ModelConfig(**changes)


def test_chunk_boundary_states_own_compact_storage():
    from glm53_flash.linear import _chunk
    q=torch.rand(1,16,2,8,requires_grad=True)
    output,state_k,state_kv=_chunk(q,q,q,q.new_zeros(1,2,8),q.new_zeros(1,2,8,8))
    for state in (state_k,state_kv):
        assert state.untyped_storage().nbytes()==state.numel()*state.element_size()
    (output.sum()+state_k.sum()+state_kv.sum()).backward()
    assert torch.isfinite(q.grad).all()


def test_combined_extensions_train_and_reload(small,tmp_path):
    config=replace(small,padding_aware_routing=True,linear_chunk_size=3,
                   linear_checkpoint_chunks=True,router_bias_rate=.01,future_heads=2)
    model=GLM53FlashFromScratch(config).train()
    x=torch.tensor([[4,5,6,7,8,0,0],[9,10,11,12,13,14,15]])
    labels=x.clone();labels[labels==0]=-100
    logits,_=model(x)
    loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)+.1*model.future_prediction_loss(labels)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.AdamW(model.parameters(),lr=.001).step();model.update_router_biases()
    model.eval();expected=model(x)[0].detach()
    save_checkpoint(model,tmp_path/'combined',{})
    restored=load_checkpoint(tmp_path/'combined',torch.device('cpu')).eval()
    assert torch.equal(expected,restored(x)[0])
