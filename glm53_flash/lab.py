"""Subclassable model facade for custom experiments and notebook inference."""
from dataclasses import replace
from pathlib import Path
from .model import ModdedGLM53FlashFromScratch,ModelConfig
from .inference import generate_text
from .runtime import load_checkpoint,save_checkpoint

class ModdedModel(ModdedGLM53FlashFromScratch):
    """Full model with configuration overrides, generation and checkpoint helpers.

    Constructor overrides create new random weights; trained checkpoint shapes
    cannot be changed in-place. Subclasses can override forward or add modules.
    """
    def __init__(self,config=None,**overrides):
        if isinstance(config,dict):config=ModelConfig(**config)
        super().__init__(replace(config or ModelConfig(),**overrides))
    def generate(self,prompt,**options):return generate_text(self,prompt,**options)
    def save(self,path,metadata=None):return save_checkpoint(self,Path(path),metadata or {})
    @classmethod
    def from_checkpoint(cls,path,device='cpu'):
        source=load_checkpoint(Path(path),device)
        if not isinstance(source,ModdedGLM53FlashFromScratch):raise ValueError('ModdedModel requires a modded checkpoint, not the pinned Vuk implementation')
        model=cls(source.config).to(device);model.load_state_dict(source.state_dict(),strict=True)
        return model.eval()
