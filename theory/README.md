# Theory notebooks by Pop Alexandru

I wrote the theory notebooks in this collection for **educational purposes**: to explain the model's building blocks in my own words and make the architecture easier to study. You are welcome to read the lessons, work through the examples, and suggest corrections or new topics.

You do not need a GPU or a training run to read them. The links below open ordinary Markdown reading pages on GitHub; the original notebooks remain available for Jupyter and Colab. Readers with access can open them now, and anyone can read them once the repository is public. Educational reuse follows the repository's [MIT license](../LICENSE), including its attribution requirement.

The original learning notes are by **Pop Alexandru**. Verification labels, placeholder notices and export headers are repository editorial additions. These notes are learning material, not claims of a frontier-model reproduction or experimentally established performance.

## Available lessons

| Topic | Read on GitHub | Original notebook |
| --- | --- | --- |
| 22M Parameters MoE Profile | [Read lesson](../docs/theory/22M_Parameters_MoE_Profile.md) | [Notebook](22M_Parameters_MoE_Profile.ipynb) |
| Explain MoE | [Read lesson](../docs/theory/Explain_MoE.md) | [Notebook](Explain_MoE.ipynb) |
| Explain RMSNorm | [Read lesson](../docs/theory/Explain_RMSNorm.md) | [Notebook](Explain_RMSNorm.ipynb) |
| Explain SwiGLU | [Read lesson](../docs/theory/Explain_SwiGLU.md) | [Notebook](Explain_SwiGLU.ipynb) |
| VisionEncoder logic | [Read lesson](../docs/theory/VisionEncoder_logic.md) | [Notebook](VisionEncoder_logic.ipynb) |
| Why Do We Obfuscate Tasks | [Read lesson](../docs/theory/Why_Do_We_Obfuscate_Tasks.md) | [Notebook](Why_Do_We_Obfuscate_Tasks.ipynb) |

The parameter-count walkthrough uses explicitly hypothetical numbers. Use [DOCUMENTATION.md](../DOCUMENTATION.md) and the actual run receipts for measured repository counts. The vision lesson benefits from first reading `glm53_flash/model.py` and `glm53_flash/vision.py`.

## Planned lessons

The following files are placeholders in Pop Alexandru's notebook series, not completed tutorials:

- [Explain Embeddings](Explain_Embeddings.ipynb)
- [Explain Linear Attention](Explain_Linear_Attention.ipynb)
- [Explain PyTorch Broadcasting](Explain_PyTorch_Broadcasting.ipynb)
- [Explain Residual Streams](Explain_Residual_Streams.ipynb)
- [Explain RoPE](Explain_RoPE.ipynb)
- [Explain Sparse Attention](Explain_Sparse_Attention.ipynb)
- [Explain Standard Attention](Explain_Standard_Attention.ipynb)
- [HyperConnection vs Mixture of Experts](HyperConnection_vs_Mixture_of_Experts.ipynb)

## Reading and contributing

Start with RMSNorm and SwiGLU, then MoE; explore task obfuscation and the optional vision path afterward. Read the parameter-count example as an arithmetic exercise. The complete model implementation and its current limitations are described in [DOCUMENTATION.md](../DOCUMENTATION.md).

Found an unclear explanation or a mathematical error? Open an issue identifying the notebook and cell, explain the correction, and include a small example where helpful. New lessons and diagrams are welcome through [CONTRIBUTING.md](../CONTRIBUTING.md). Preserve the author attribution and distinguish original notes from later contributions.

## Reading versus execution

These notebooks do not require a cloned repository for reading. Most contain only explanatory Markdown; the vision notebook includes contextual code snippets, not a standalone runnable program. For actual training use the experiment notebooks with the repository source available. See [VS Code/Jupyter setup](../IDE_GUIDE.md) or [Colab setup](../COLAB.md).
