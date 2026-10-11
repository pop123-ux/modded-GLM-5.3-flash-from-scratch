# Explain MoE

**Author: Pop Alexandru** · Educational theory notebook

I wrote this notebook to share my understanding of the ideas behind the model. You are welcome to read it for learning, follow the examples, and suggest corrections or additions.

The original learning notes are preserved below. Repository-specific verification labels and packaging notes are editorial additions; any hypothetical counts are teaching examples rather than benchmark measurements.


[Open the original notebook](../../theory/Explain_MoE.ipynb) · [Notebook index](../../theory/README.md)

This reading page mirrors the notebook cells. Code is shown for study; it has not been executed as part of this export.


## Mixture of Experts ##
- Mixture of Experts (MoE) represents an advanced architectural design for LLMs that makes them much larger, smarter, and faster, while keeping the computational cost low.

- A MoE model is broken up into smaller, specialized subnetworks called "Experts".

## Why is it shifting to be a standard for modern LLMs? ##
- In a MoE model, the inference forward method changes into a dynamic sorting and routing hub.

- Instead of processing every token through one massive network, the model acts as a smart driver that directs different tokens to different so-called "Experts" before gathering their answers.

## How does the forward pass change? ##
- When a token enters an MoE layer, it does not immediately go to a processing network. Instead, it hits a lightweight module called the Router (or Gatekeeper).

- The Router takes the token's current vector representation and performs a mathematical computation to determine which experts in the pool are best suited to handle this specific piece of data.

- For eg., if the token is "Python", the router might decide Expert #3 (coding) and Expert #5 (syntax) should handle it.

## How does the router select the expert to handle the specific task? ##
- The router outputs a list of probabilities for all available experts. The forward method is configured to pick the top choices-usually top 1 or top 2 experts (known as Top-1 or Top-2 routing).

- **To note:** If an expert doesn't make the top cut for that specific token, it is completely ignored. This means its weights are never loaded into the active compute cycle, saving massive amounts of processing power.

## How does the router split and group the tokens? ##
- The model groups all tokens from the sentence that chose Expert #1 into one batch.

- It groups all tokens that chose Expert #2 into another batch, and so on.

- Tokens are processed by their assigned experts completely independently and in parallel.

## How does it weigh the Expert Answers? ##
- Because at the time of designing this project, Top-2 routing is common, a single token will often get two separate answers back from two different experts. The forward method doesn't just blindly blend them; it multiplies each expert's output by the confidence score that the Router gave it in Step 2.

- For eg., If Expert #3 was 80% confident and Expert #5 was 20% confident, their respective outputs are scaled by 0.8 and 0.2.

## How does this change the computation the model does? ##
- Once all the activated experts finish their calculations, the forward method executes a gather operation. It takes the processed tokens from all the different expert branches and reconstructs them back into the exact original sequence.

- The sequence then moves forward to the next layer of the transformer, where the exact same routing process happens all over again.

# How does the architecture prevent all the experts from learning the same thing? ##
- During training, researchers have added a specific mathematical penalty to the model's loss function called the **load balancing loss**.

-> How does it work: The model is penalized if the router sends too many tokens to one expert while leaving others empty.

-> The Result: The router is forced to give under-utilized experts a chance. If Expert #A is overwhelmed with data, the router is mathematically penalized and starts routing some text to Expert #B and Expert #C, forcing them to learn how to process information.

## Capacity Limiting (Token Dropping) ##
- During training, each expert is given a strict **capacity limit**-a maximum number of tokens it is allowed to process in a single batch.

-> How it works: If a popular expert hits its maximum capacity ceiling, it completely shuts its doors for the rest of that training step. Any remaining tokens assigned to it are either dropped or forced onto the second-choice experts.

-> The Result: This bottleneck forces the secondary experts to step up and learn the concepts, preventing a single "genius" expert from hogging all the training data.

## The initial findings were actually beautiful ##
- Once training is complete, researchers analyze the experts to see what they learned. They found that they commonly specialize in abstract linguistic concepts:

1. **Syntactic Experts:** Focus purely on punctuation, grammar, pronouns, or structural sentence transitions.

2. **Semantic Experts:** Focus on specific domains like computer code, mathematical symbols, or localized languages (e.g., German syntax).

3. **Contextual Experts:** Trigger only when a sentence requires abstract reasoning or long-range factual recall.
