"""Full-context autoregressive inference, without a KV cache."""
from dataclasses import dataclass, asdict
import math
import torch
from .tokenizer import ByteTokenizer

@dataclass
class GenerationResult:
    prompt: str
    completion: str
    token_ids: list[int]
    stopped_on_eos: bool
    @property
    def text(self):return self.prompt+self.completion
    def to_dict(self):return asdict(self)|dict(text=self.text,generated_tokens=len(self.token_ids))

@torch.inference_mode()
def generate_text(model,prompt,*,max_new_tokens=64,temperature=0.,top_k=0,top_p=1.,seed=42,tokenizer=None):
    """Greedy or seeded sampling. Preserve training mode and the global RNG.

    Sampling uses a local CPU generator (including for GPU models). Context is
    measured in bytes; prompt plus requested cap must fit without truncation.
    """
    if not isinstance(prompt,str):raise TypeError('prompt must be a string')
    if not isinstance(max_new_tokens,int) or max_new_tokens<1:raise ValueError('max_new_tokens must be a positive integer')
    if not math.isfinite(temperature) or temperature<0:raise ValueError('temperature must be finite and nonnegative')
    if not isinstance(top_k,int) or top_k<0:raise ValueError('top_k must be a nonnegative integer')
    if not math.isfinite(top_p) or not 0<top_p<=1:raise ValueError('top_p must be in (0,1]')
    tok=tokenizer or ByteTokenizer();ids=tok.encode(prompt,bos=True)
    if len(ids)+max_new_tokens>model.config.max_sequence_length:raise ValueError('prompt plus generation cap exceeds model context')
    device=next(model.parameters()).device;generator=torch.Generator(device='cpu').manual_seed(seed)
    generated=[];previous=model.training;model.eval()
    try:
        for _ in range(max_new_tokens):
            logits,_=model(torch.tensor([ids+generated],device=device));values=logits[0,-1].float().clone()
            values[[tok.pad_id,tok.bos_id,tok.sep_id]]=-torch.inf
            if not bool(torch.isfinite(values).any()) or bool(torch.isnan(values).any()) or bool(torch.isposinf(values).any()):raise FloatingPointError('invalid generation logits')
            if temperature==0:token=int(values.argmax())
            else:
                values,indices=((values-values.max())/temperature).sort(descending=True)
                if top_k:values[min(top_k,len(values)):]=-torch.inf
                probabilities=values.softmax(-1)
                if top_p<1:
                    values[probabilities.cumsum(-1)-probabilities>=top_p]=-torch.inf
                    probabilities=values.softmax(-1)
                chosen=int(torch.multinomial(probabilities.cpu(),1,generator=generator));token=int(indices[chosen])
            generated.append(token)
            if token==tok.eos_id:break
    finally:model.train(previous)
    return GenerationResult(prompt,tok.decode(generated),generated,bool(generated and generated[-1]==tok.eos_id))
