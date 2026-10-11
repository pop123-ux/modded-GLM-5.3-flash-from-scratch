# Explain SwiGLU

**Author: Pop Alexandru** · Educational theory notebook

I wrote this notebook to share my understanding of the ideas behind the model. You are welcome to read it for learning, follow the examples, and suggest corrections or additions.

The original learning notes are preserved below. Repository-specific verification labels and packaging notes are editorial additions; any hypothetical counts are teaching examples rather than benchmark measurements.


[Open the original notebook](../../theory/Explain_SwiGLU.ipynb) · [Notebook index](../../theory/README.md)

This reading page mirrors the notebook cells. Code is shown for study; it has not been executed as part of this export.


# Comprehensive Guide to SwiGLU Activation & Projections in LLMs

This notebook breaks down the **SwiGLU** (*Swish Gated Linear Unit*) activation mechanism — a widely used Feed-Forward Network (**FFN / MLP**) design in modern Large Language Models such as **LLaMA, Mistral, and Gemma**.

---

# 1. Theoretical Foundation

## 1.1 Swish / SiLU

**Swish** is a smooth nonlinear activation function.

Unlike ReLU, which completely sets negative values to zero, Swish preserves some information for moderately negative inputs while maintaining a smooth gradient.

The general Swish function is:

$$
\mathrm{Swish}_{\beta}(x)
=
x \cdot \sigma(\beta x)
$$

where the sigmoid function is:

$$
\sigma(x)
=
\frac{1}{1 + e^{-x}}
$$

In most modern LLM implementations:

$$
\beta = 1
$$

Therefore:

$$
\mathrm{Swish}(x)
=
x \cdot \sigma(x)
$$

This is mathematically identical to the **SiLU** activation:

$$
\mathrm{SiLU}(x)
=
x \cdot \sigma(x)
$$

So, in practice:

$$
\boxed{
\mathrm{Swish}(x)
=
\mathrm{SiLU}(x)
}
$$

---

## 1.2 GLU — Gated Linear Unit

A **Gated Linear Unit** uses two parallel transformations of the same input.

One branch contains the information being transformed, while another branch controls how strongly individual features are allowed to pass through.

A simplified GLU can be written as:

$$
\mathrm{GLU}(x)
=
\sigma(xW + b)
\odot
(xV + c)
$$

where:

- $W$ and $V$ are learned weight matrices,
- $b$ and $c$ are optional bias vectors,
- $\sigma$ is the sigmoid activation,
- $\odot$ represents the **Hadamard product**, meaning element-wise multiplication.

---

# 2. The SwiGLU Equation

SwiGLU combines the **gating structure of GLU** with the **SiLU / Swish activation**.

The core operation is:

$$
\boxed{
\mathrm{SwiGLU}(x)
=
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
}
$$

The complete Feed-Forward Network adds a final down projection:

$$
\boxed{
\mathrm{FFN}(x)
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

The three learned projection matrices are:

$$
W_{\mathrm{gate}}
$$

$$
W_{\mathrm{up}}
$$

$$
W_{\mathrm{down}}
$$

These correspond to:

1. **Gate projection**
2. **Up projection**
3. **Down projection**

---

# 3. Architecture and Data Flow

A traditional Transformer MLP typically performs:

$$
d_{\mathrm{model}}
\rightarrow
d_{\mathrm{ff}}
\rightarrow
d_{\mathrm{model}}
$$

SwiGLU splits the expansion stage into **two parallel projections**.

```text
Input Tensor
(B, L, d_model)

                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│ Gate Project. │ │  Up Project.  │
│    W_gate     │ │     W_up      │
└───────────────┘ └───────────────┘
        │               │
        ▼               │
┌───────────────┐       │
│     SiLU      │       │
│    / Swish    │       │
└───────────────┘       │
        │               │
        └───────┬───────┘
                │
                ▼
      Element-wise Multiply
               ⊙
                │
                ▼
       Combined Hidden State
           (B, L, d_ff)
                │
                ▼
       ┌────────────────┐
       │ Down Projection│
       │     W_down     │
       └────────────────┘
                │
                ▼
           Output Tensor
        (B, L, d_model)
```

Mathematically, the forward pass can be broken into four steps.

### Step 1 — Gate projection

$$
g
=
xW_{\mathrm{gate}}
$$

### Step 2 — Up projection

$$
u
=
xW_{\mathrm{up}}
$$

### Step 3 — Gating

$$
h
=
\mathrm{SiLU}(g)
\odot
u
$$

### Step 4 — Down projection

$$
y
=
hW_{\mathrm{down}}
$$

Combining everything:

$$
\boxed{
y
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

---

# 4. Deep Dive Into the Three Projections

## 4.1 Gate Projection — `gate_proj`

The gate projection is:

$$
g
=
xW_{\mathrm{gate}}
$$

It transforms the input from the model hidden dimension:

$$
d_{\mathrm{model}}
$$

into the FFN intermediate dimension:

$$
d_{\mathrm{ff}}
$$

Therefore:

$$
d_{\mathrm{model}}
\rightarrow
d_{\mathrm{ff}}
$$

The output is then passed through SiLU:

$$
\hat{g}
=
\mathrm{SiLU}(g)
$$

or equivalently:

$$
\boxed{
\hat{g}
=
\mathrm{SiLU}(xW_{\mathrm{gate}})
}
$$

### Function

The gate projection determines how strongly individual intermediate features contribute to the final FFN representation.

Because SiLU can produce:

- values close to zero,
- positive values,
- small negative values,
- values larger than one,

the gate can smoothly:

- suppress features,
- preserve features,
- amplify features,
- reverse the sign of features in some cases.

### Intuition

Think of the gate projection as a **dynamic feature controller**.

For every token and every intermediate feature, the network learns something conceptually similar to:

> **How strongly should this feature influence the result?**

---

## 4.2 Up Projection — `up_proj`

The up projection is:

$$
u
=
xW_{\mathrm{up}}
$$

Like the gate projection, it transforms:

$$
d_{\mathrm{model}}
\rightarrow
d_{\mathrm{ff}}
$$

However, this branch does **not** receive an activation function directly.

### Function

The up projection creates a learned representation of the input in the larger FFN feature space.

It runs in parallel with the gate projection.

### Intuition

A useful mental model is:

- **Up projection** → produces candidate features.
- **Gate projection** → determines how strongly those features should contribute.

The two branches meet through element-wise multiplication:

$$
\boxed{
h
=
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
}
$$

This multiplication is the central mechanism of SwiGLU.

---

# 5. The Element-Wise Merge

Suppose the activated gate produces:

$$
\mathrm{SiLU}(g)
=
[0.2,\ 1.4,\ -0.1,\ 0.8]
$$

and the up projection produces:

$$
u
=
[5,\ 2,\ 7,\ -3]
$$

SwiGLU multiplies them element by element:

$$
[0.2,\ 1.4,\ -0.1,\ 0.8]
\odot
[5,\ 2,\ 7,\ -3]
$$

giving:

$$
[1,\ 2.8,\ -0.7,\ -2.4]
$$

Therefore each intermediate feature is independently controlled by its corresponding gate value.

In scalar form:

$$
h_i
=
\mathrm{SiLU}(g_i)
\cdot
u_i
$$

for every intermediate feature:

$$
i = 1,\ldots,d_{\mathrm{ff}}
$$

---

# 6. Down Projection — `down_proj`

After the gating operation, the hidden representation still has dimensionality:

$$
d_{\mathrm{ff}}
$$

The down projection transforms it back into the Transformer hidden dimension:

$$
d_{\mathrm{ff}}
\rightarrow
d_{\mathrm{model}}
$$

The operation is:

$$
y
=
hW_{\mathrm{down}}
$$

Substituting the SwiGLU hidden state:

$$
\boxed{
y
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

### Function

The down projection takes the large intermediate FFN representation and maps it back into the dimensionality of the Transformer's **residual stream**.

### Intuition

The expansion into $d_{\mathrm{ff}}$ gives the network a larger feature space in which to manipulate the representation.

The down projection then:

1. mixes those intermediate features,
2. compresses them,
3. returns them to dimension $d_{\mathrm{model}}$.

The result can then be combined with the residual connection of the Transformer block.

---

# 7. Tensor Shape Breakdown

Let:

- $B$ = batch size
- $L$ = sequence length
- $d_{\mathrm{model}}$ = Transformer hidden dimension
- $d_{\mathrm{ff}}$ = FFN intermediate dimension

The input tensor is:

$$
X
\in
\mathbb{R}^{B \times L \times d_{\mathrm{model}}}
$$

For example:

```text
Batch Size        B = 8
Sequence Length   L = 2048
Model Dimension   d_model = 4096
```

Then the input shape is:

```text
(8, 2048, 4096)
```

---

## 7.1 Gate Projection Shape

The gate matrix is:

$$
W_{\mathrm{gate}}
\in
\mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}
$$

Therefore:

$$
XW_{\mathrm{gate}}
\in
\mathbb{R}^{B \times L \times d_{\mathrm{ff}}}
$$

After SiLU:

$$
\mathrm{SiLU}(XW_{\mathrm{gate}})
\in
\mathbb{R}^{B \times L \times d_{\mathrm{ff}}}
$$

---

## 7.2 Up Projection Shape

The up-projection matrix is:

$$
W_{\mathrm{up}}
\in
\mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}
$$

Therefore:

$$
XW_{\mathrm{up}}
\in
\mathbb{R}^{B \times L \times d_{\mathrm{ff}}}
$$

---

## 7.3 Element-Wise Merge Shape

Both branches have identical shapes:

$$
(B,L,d_{\mathrm{ff}})
$$

Therefore they can be multiplied element-wise:

$$
H
=
\mathrm{SiLU}(XW_{\mathrm{gate}})
\odot
(XW_{\mathrm{up}})
$$

The output shape remains:

$$
H
\in
\mathbb{R}^{B \times L \times d_{\mathrm{ff}}}
$$

---

## 7.4 Down Projection Shape

The down-projection matrix is:

$$
W_{\mathrm{down}}
\in
\mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}
$$

Therefore:

$$
HW_{\mathrm{down}}
\in
\mathbb{R}^{B \times L \times d_{\mathrm{model}}}
$$

The original Transformer hidden dimension is restored.

---

# 8. Dimensions Table

| Operation | Input Shape | Weight Shape | Output Shape | Activation |
|---|---|---|---|---|
| **Gate Projection** | $(B,L,d_{\mathrm{model}})$ | $(d_{\mathrm{model}},d_{\mathrm{ff}})$ | $(B,L,d_{\mathrm{ff}})$ | SiLU |
| **Up Projection** | $(B,L,d_{\mathrm{model}})$ | $(d_{\mathrm{model}},d_{\mathrm{ff}})$ | $(B,L,d_{\mathrm{ff}})$ | None |
| **Element-wise Merge** | Two $(B,L,d_{\mathrm{ff}})$ tensors | N/A | $(B,L,d_{\mathrm{ff}})$ | Multiplication |
| **Down Projection** | $(B,L,d_{\mathrm{ff}})$ | $(d_{\mathrm{ff}},d_{\mathrm{model}})$ | $(B,L,d_{\mathrm{model}})$ | None |

---

# 9. Traditional Transformer MLP

A traditional Transformer Feed-Forward Network typically looks like:

```text
Input
  │
  ▼
Up Projection
  │
  ▼
Activation
  │
  ▼
Down Projection
  │
  ▼
Output
```

Mathematically:

$$
y
=
\phi(xW_{\mathrm{up}})
W_{\mathrm{down}}
$$

where $\phi$ might be:

- ReLU,
- GELU,
- another nonlinear activation.

The parameterized part contains two large matrices:

$$
W_{\mathrm{up}}
$$

and:

$$
W_{\mathrm{down}}
$$

---

# 10. SwiGLU MLP

SwiGLU instead performs:

```text
                    ┌─────────────────┐
                    │ Gate Projection │
                    │     W_gate      │
                    └────────┬────────┘
                             │
                             ▼
Input ───────────────────►  SiLU
 │                           │
 │                           │
 │                           ▼
 │                         Gate
 │                           │
 │                           ▼
 │                       ┌───────┐
 │                       │   ⊙   │
 │                       └───┬───┘
 │                           ▲
 │                           │
 │                           │
 │                  ┌────────┴────────┐
 └─────────────────►│  Up Projection │
                    │      W_up      │
                    └─────────────────┘

                             │
                             ▼
                    Combined Hidden State
                             │
                             ▼
                    ┌─────────────────┐
                    │ Down Projection │
                    │     W_down      │
                    └────────┬────────┘
                             │
                             ▼
                           Output
```

Mathematically:

$$
\boxed{
y
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

The key architectural difference is:

> A traditional MLP creates one expanded representation and applies an activation to it, while SwiGLU creates two expanded representations and uses one to dynamically gate the other.

---

# 11. Why SwiGLU Uses Three Projections

A conventional MLP contains two large weight matrices:

$$
W_1
\in
\mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}
$$

and:

$$
W_2
\in
\mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}
$$

Ignoring biases, the parameter count is approximately:

$$
P_{\mathrm{traditional}}
=
2d_{\mathrm{model}}d_{\mathrm{ff}}
$$

A common traditional FFN width is:

$$
d_{\mathrm{ff}}
=
4d_{\mathrm{model}}
$$

Substituting:

$$
P_{\mathrm{traditional}}
=
2d_{\mathrm{model}}
\left(
4d_{\mathrm{model}}
\right)
$$

Therefore:

$$
\boxed{
P_{\mathrm{traditional}}
=
8d_{\mathrm{model}}^2
}
$$

---

# 12. SwiGLU Parameter Count

SwiGLU has three large matrices:

$$
W_{\mathrm{gate}}
$$

$$
W_{\mathrm{up}}
$$

$$
W_{\mathrm{down}}
$$

Each contributes approximately:

$$
d_{\mathrm{model}}d_{\mathrm{ff}}
$$

parameters.

Therefore:

$$
P_{\mathrm{SwiGLU}}
=
3d_{\mathrm{model}}d_{\mathrm{ff}}
$$

To keep approximately the same parameter budget as a conventional $4d_{\mathrm{model}}$ FFN:

$$
3d_{\mathrm{model}}d_{\mathrm{ff}}
\approx
8d_{\mathrm{model}}^2
$$

Divide both sides by $d_{\mathrm{model}}$:

$$
3d_{\mathrm{ff}}
\approx
8d_{\mathrm{model}}
$$

Therefore:

$$
\boxed{
d_{\mathrm{ff}}
\approx
\frac{8}{3}d_{\mathrm{model}}
}
$$

Numerically:

$$
\frac{8}{3}
\approx
2.67
$$

so:

$$
\boxed{
d_{\mathrm{ff}}
\approx
2.67d_{\mathrm{model}}
}
$$

This is why SwiGLU architectures often use a narrower intermediate dimension than the classic:

$$
4d_{\mathrm{model}}
$$

rule.

---

# 13. Why the Intermediate Dimension Is Often Rounded

Real LLM implementations do not necessarily use exactly:

$$
\frac{8}{3}d_{\mathrm{model}}
$$

Instead, the result is frequently rounded to dimensions that are more convenient for GPU hardware.

For example:

$$
d_{\mathrm{model}}
=
4096
$$

would theoretically give:

$$
d_{\mathrm{ff}}
=
\frac{8}{3}(4096)
$$

$$
d_{\mathrm{ff}}
\approx
10922.67
$$

A real architecture may choose a nearby hardware-friendly value such as:

```text
10752
11008
11264
```

depending on the model design.

These dimensions are often selected to improve:

- GPU utilization,
- tensor-core efficiency,
- memory alignment,
- distributed-training efficiency.

---

# 14. Why Gating Is Powerful

Consider the SwiGLU hidden representation:

$$
h
=
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
$$

The value of every output feature depends on **two learned transformations** of the input.

For feature $i$:

$$
h_i
=
\mathrm{SiLU}(g_i)
u_i
$$

This means the network can learn interactions such as:

```text
Candidate feature exists
          │
          ▼
   Is it useful here?
          │
     ┌────┴────┐
     │         │
    Yes        No
     │         │
     ▼         ▼
  Preserve   Suppress
```

The gating mechanism therefore provides a form of **input-dependent feature modulation**.

---

# 15. Why SiLU Works Well as the Gate

The SiLU function is:

$$
\mathrm{SiLU}(x)
=
x\sigma(x)
$$

For very negative values:

$$
\mathrm{SiLU}(x)
\rightarrow
0
$$

For values near zero, the function changes smoothly.

For large positive values:

$$
\mathrm{SiLU}(x)
\approx
x
$$

Therefore the gate can smoothly transition between:

```text
Suppress
   ↓
Near Zero
   ↓
Preserve
   ↓
Amplify
```

rather than behaving like a strict binary switch.

---

# 16. SwiGLU Is Not a Binary Gate

The word **gate** can be misleading.

SwiGLU does not normally produce only:

$$
0
$$

or:

$$
1
$$

Instead:

$$
\mathrm{SiLU}(xW_{\mathrm{gate}})
$$

can produce a continuous range of values.

For example:

$$
[-0.12,\ 0.02,\ 0.45,\ 1.3,\ 4.8]
$$

This means the gate performs **continuous modulation**, rather than simple ON/OFF switching.

---

# 17. Minimal PyTorch Implementation

A minimal implementation is:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.up_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.down_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False
        )

    def forward(self, x):

        gate = self.gate_proj(x)

        up = self.up_proj(x)

        hidden = F.silu(gate) * up

        output = self.down_proj(hidden)

        return output
```

The most important line is:

```python
hidden = F.silu(gate) * up
```

which implements:

$$
\boxed{
h
=
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
}
$$

The final line:

```python
output = self.down_proj(hidden)
```

implements:

$$
y
=
hW_{\mathrm{down}}
$$

Together:

$$
\boxed{
y
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

---

# 18. PyTorch Shape Example

Consider:

```python
B = 2
L = 128
d_model = 4096
d_ff = 11008
```

The input might have shape:

```python
x.shape
```

```text
torch.Size([2, 128, 4096])
```

After the gate projection:

```python
gate = self.gate_proj(x)
```

the shape becomes:

```text
torch.Size([2, 128, 11008])
```

After the up projection:

```python
up = self.up_proj(x)
```

the shape is also:

```text
torch.Size([2, 128, 11008])
```

The gated hidden state:

```python
hidden = F.silu(gate) * up
```

still has shape:

```text
torch.Size([2, 128, 11008])
```

Finally:

```python
output = self.down_proj(hidden)
```

returns:

```text
torch.Size([2, 128, 4096])
```

So the complete dimensional flow is:

```text
(B, L, 4096)
       │
       ├───────────────┐
       ▼               ▼
(B, L, 11008)   (B, L, 11008)
   Gate               Up
       │               │
      SiLU             │
       │               │
       └───────⊙───────┘
               │
               ▼
       (B, L, 11008)
               │
             Down
               │
               ▼
        (B, L, 4096)
```

---

# 19. Compact Mathematical Derivation

Starting with:

$$
x
\in
\mathbb{R}^{d_{\mathrm{model}}}
$$

compute the gate:

$$
g
=
xW_{\mathrm{gate}}
$$

Compute the candidate features:

$$
u
=
xW_{\mathrm{up}}
$$

Activate the gate:

$$
a
=
\mathrm{SiLU}(g)
$$

Combine both paths:

$$
h
=
a
\odot
u
$$

Project back:

$$
y
=
hW_{\mathrm{down}}
$$

Substitution gives:

$$
y
=
\left(
\mathrm{SiLU}(g)
\odot
u
\right)
W_{\mathrm{down}}
$$

Then substituting $g$ and $u$:

$$
\boxed{
y
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

---

# 20. Traditional MLP vs. SwiGLU

| Property | Traditional MLP | SwiGLU MLP |
|---|---|---|
| Input projections | 1 | 2 |
| Total major projections | 2 | 3 |
| Gating mechanism | No | Yes |
| Typical activation | GELU / ReLU | SiLU |
| Intermediate representations | 1 | 2 |
| Element-wise feature gating | No | Yes |
| Typical classic width | $\approx 4d_{\mathrm{model}}$ | Often around $\frac{8}{3}d_{\mathrm{model}}$ before architecture-specific rounding |

---

# 21. Final Mental Model

A useful way to remember SwiGLU is:

```text
                  WHAT FEATURES?
                       │
                       ▼
Input ─────────► Up Projection
                       │
                       │
                       ▼
                  Candidate Features
                       │
                       │
                       ▼
                    Multiply
                       ▲
                       │
                       │
                  Activated Gate
                       ▲
                       │
Input ───────► Gate Projection
                       │
                       ▼
                      SiLU

                       │
                       ▼
               Gated Hidden State
                       │
                       ▼
                Down Projection
                       │
                       ▼
                     Output
```

An approximate conceptual interpretation is:

| Component | Mental Model |
|---|---|
| **Up Projection** | What candidate features should be produced? |
| **Gate Projection** | How strongly should each feature participate? |
| **SiLU** | Smoothly modulate the gate values |
| **Element-wise multiplication** | Apply the gate to each corresponding feature |
| **Down Projection** | Mix and return the features to the residual-stream dimension |

---

# 22. One-Line Summary

The entire SwiGLU MLP can be summarized as:

$$
\boxed{
\mathrm{SwiGLUFFN}(x)
=
\left[
\mathrm{SiLU}(xW_{\mathrm{gate}})
\odot
(xW_{\mathrm{up}})
\right]
W_{\mathrm{down}}
}
$$

In words:

> **Project the input twice, activate one projection with SiLU, use it to gate the other projection element-by-element, and project the resulting representation back to the model dimension.**

---

# 23. Ultimate Intuition

The standard MLP asks:

> **What features should I compute from this token?**

SwiGLU effectively adds another learned question:

> **Given this token and its current representation, how strongly should each of those features matter?**

This extra gating mechanism is what makes SwiGLU more expressive than simply applying a nonlinear activation to a single projection.

The architecture can be remembered as:

$$
\boxed{
\mathrm{Input}
\rightarrow
\begin{cases}
\mathrm{Gate} \rightarrow \mathrm{SiLU} \\
\mathrm{Up}
\end{cases}
\rightarrow
\mathrm{Multiply}
\rightarrow
\mathrm{Down}
\rightarrow
\mathrm{Output}
}
$$
