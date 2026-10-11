# Why Do We Obfuscate Tasks

**Author: Pop Alexandru** · Educational theory notebook

I wrote this notebook to share my understanding of the ideas behind the model. You are welcome to read it for learning, follow the examples, and suggest corrections or additions.

The original learning notes are preserved below. Repository-specific verification labels and packaging notes are editorial additions; any hypothetical counts are teaching examples rather than benchmark measurements.


[Open the original notebook](../../theory/Why_Do_We_Obfuscate_Tasks.ipynb) · [Notebook index](../../theory/README.md)

This reading page mirrors the notebook cells. Code is shown for study; it has not been executed as part of this export.


## Obfuscation & Dynamic Entry Point Definition

Inside the `tasks.py` file, the **_name** function creates a obfuscated version of the original function name.

This is standard practice in modern LLM benchmark testing.

Let's say an advanced LLM model was given a dataset to train on, and afterwards it is decided to be evaluate on a benchmark dataset. If a benchmark always uses standard names (e.g.: `def fizzbuzz(n)`; `def factorial(n)`), the LLM might just regurgitate memorized code from the training data.

So by scrambling the entry point to a completely randomized string, it forces the AI model to actually parse the instructions and reason about what code to write
