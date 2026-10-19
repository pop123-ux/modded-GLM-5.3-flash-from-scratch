"""Scaled GLM-5.3-Flash architecture implemented from scratch."""

from .model import GLM53FlashFromScratch, ModelConfig
from .tokenizer import ByteTokenizer
from .vision import (
    DirectPatchImageEncoder,
    DirectPatchVisionLanguageModel,
    ImageTokenIds,
    MiniGLMVisionEncoder,
    MiniVisionConfig,
    MultimodalProjector,
    PatchImageEncoder,
    RGBPatchEmbedding,
    SpatialMerger2x2,
    VisionLanguageModel,
    VisionTransformerBlock,
    answer_only_labels,
    build_multimodal_input_ids,
)

__all__ = [
    "ByteTokenizer",
    "DirectPatchImageEncoder",
    "DirectPatchVisionLanguageModel",
    "GLM53FlashFromScratch",
    "ImageTokenIds",
    "MiniGLMVisionEncoder",
    "MiniVisionConfig",
    "ModelConfig",
    "MultimodalProjector",
    "PatchImageEncoder",
    "RGBPatchEmbedding",
    "SpatialMerger2x2",
    "VisionLanguageModel",
    "VisionTransformerBlock",
    "answer_only_labels",
    "build_multimodal_input_ids",
]

from .model import ModdedGLM53FlashFromScratch
from .inference import generate_text, GenerationResult
from .lab import ModdedModel
__all__ += ['ModdedGLM53FlashFromScratch','ModdedModel','generate_text','GenerationResult']
