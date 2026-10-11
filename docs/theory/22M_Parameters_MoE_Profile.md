# 22M Parameters MoE Profile

**Author: Pop Alexandru** · Educational theory notebook

I wrote this notebook to share my understanding of the ideas behind the model. You are welcome to read it for learning, follow the examples, and suggest corrections or additions.

The original learning notes are preserved below. Repository-specific verification labels and packaging notes are editorial additions; any hypothetical counts are teaching examples rather than benchmark measurements.


[Open the original notebook](../../theory/22M_Parameters_MoE_Profile.ipynb) · [Notebook index](../../theory/README.md)

This reading page mirrors the notebook cells. Code is shown for study; it has not been executed as part of this export.


# Hypothetical parameter-count walkthrough
The 22M/16M values below are assumed teaching numbers, not measured counts of this repository. The repaired default has 12,625,740 total parameters and an active-per-token architectural estimate of 7,907,148. Active parameters are not an exact FLOP or memory estimate.


### Let's assume our **ModelConfig** produces the following structural distribution:

- Total parameters (total): 22,000,000 (22M)

- Isolated expert parameters (routed_total): 16,000,000 (16M)

- Experts configuration: experts = 8, top_k = 2 (The router picks 2 out of 8 experts per token)

### Using the mathematical logic implemented inside the `param_count()` method:

#### 1. Isolate the Shared Dense Backbone Parameters
We subtract the entire expert pool from the total model footprint to isolate the baseline infrastructure (Embeddings, Attention layers, HyperConnection logits, and RMSNorms) that are *always* executed for every single token:

$$ \text{Dense Backbone} = \text{total} - \text{routed\_total} $$

$$ 22,000,000 - 16,000,000 = 6,000,000 \text{ (6M dense parameters)} $$

#### 2. Calculate the Active Expert Parameters
We compute the size of only the experts that are actively processing a token at any given moment (2 out of 8 experts, which scales the total expert pool down to $ \frac{1}{4} $):

$$ \text{Active Experts} = \text{routed\_total} \times \frac{\text{top\_k}}{\text{experts}} $$

$$ 16,000,000 \times \frac{2}{8} = 4,000,000 \text{ (4M active expert parameters)} $$

#### 3. Sum for the Final Per-Token Active Estimate
Combining the non-negotiable dense backbone footprint with the active expert track gives us the actual computational footprint expended per token:

$$ \text{Active Estimate} = 6,000,000 + 4,000,000 = 10,000,000 \text{ (10M parameters)} $$


**Interpretation:** Sparse routing activates a fraction of the expert parameter pool per token. The dense backbone and shared expert still run for every token. Actual speed and memory must be measured; they are not determined by the active-parameter count alone.
