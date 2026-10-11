# Explain RMSNorm

**Author: Pop Alexandru** · Educational theory notebook

I wrote this notebook to share my understanding of the ideas behind the model. You are welcome to read it for learning, follow the examples, and suggest corrections or additions.

The original learning notes are preserved below. Repository-specific verification labels and packaging notes are editorial additions; any hypothetical counts are teaching examples rather than benchmark measurements.


[Open the original notebook](../../theory/Explain_RMSNorm.ipynb) · [Notebook index](../../theory/README.md)

This reading page mirrors the notebook cells. Code is shown for study; it has not been executed as part of this export.


**Root Mean Square Normalization (RMSNorm)** is an alternative to traditional LayerNor. It scales activation vectors (arrays of numbers that represent the numerical outputs of a layaer of artificial neurons) using only the root-mean-square value, entirely skipping the mean-centering step.

- This reduction in mathematical computations makes it faster and more memory-efficient, and it is now standard in large language models like GLM.

$$y_i = \frac{x_i}{\text{RMS}(x)} * \gamma_i, \quad \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}$$

- \(y_{i}\): The final normalized output value for the \(i\)-th feature in the vector.

- \(x_{i}\): The raw input value of the \(i\)-th feature before normalization.

- \(\text{RMS}(x)\): The Root Mean Square value of the entire input vector. Dividing \(x_{i}\) by this value rescales the vector elements to a stable, unit-variance-like scale.

- \(\gamma _{i}\): A learnable weight parameter (scale factor) unique to each feature channel. It allows the model to scale up or down specific dimensions during training.
