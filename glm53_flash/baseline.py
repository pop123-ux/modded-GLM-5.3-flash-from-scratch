"""Load the exact vendored Vuk model without replacing the active package."""
import importlib.util
import sys
from pathlib import Path

def upstream_module():
    name = '_pinned_vuk_glm_model'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1]/'baselines/upstream/glm53_flash/model.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

def build_baseline(config):
    module = upstream_module()
    accepted = module.ModelConfig.__dataclass_fields__
    config = module.ModelConfig(**{k:v for k,v in config.to_dict().items() if k in accepted})
    return module.GLM53FlashFromScratch(config)
