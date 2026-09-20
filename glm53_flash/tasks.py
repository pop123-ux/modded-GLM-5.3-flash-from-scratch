"""Deterministic synthetic coding corpus and frozen executable evaluations."""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Family:
    name: str
    arguments: str
    descriptions: tuple[str, ...]
    body: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]
    
@dataclass(frozen=True)
class CodingTask:
    task_id: str
    family: str
    entry_point: str
    prompt: str
    reference_completion: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]
    
    @property
    def reference_source(self) -> str:
        return self.prompt + self.reference_completion
    
FAMILIES = (
    Family(
        "increment", 
        "x", 
        ("Return x plus one.", "Increase x by one.", "Sum x and one."), 
        "\n    return x + 1\n", 
        (((-3,), -2), ((0,), 1), ((7,), 8))
    ),
    Family(
        "double", 
        "x", 
        ("Return two times x.", "Double the input.", "Compute the product of x and 2."), 
        "\n    return x * 2\n", 
        (((-4,), -8), ((0,), 0), ((6,), 12))
    ),
    Family(
        "square", 
        "x", 
        ("Return x squared.", "Multiply x by itself.", "Compute the product of x and itself."), 
        "\n    return x * x\n", 
        (((-4,), 16), ((0,), 0), ((5,), 25))
    ),
    Family(
        "absolute", 
        "x", 
        ("Return the absolute value of x.", "Make a negative x positive.", "Compute |x|."), 
        "\n    return -x if x < 0 else x\n", 
        (((-7,), 7), ((0,), 0), ((9,), 9))
    ),
    Family(
        "nonnegative", 
        "x", 
        ("Clamp x to be at least zero.", "Return zero when x is negative.", "Compute f:R -> R+; f(x) = x."), 
        "\n    return x if x > 0 else 0\n", 
        (((-5,), 0), ((0,), 0), ((8,), 8))
    ),
    Family(
        "even", 
        "x", 
        ("Return whether x is even.", "Check divisibility by two."), 
        "\n    return x % 2 == 0\n", 
        (((-2,), True), ((-3,), False), ((0,), True), ((8,), True))
    ),
    Family(
        "reverse", 
        "text", 
        ("Return text in reverse order.", "Reverse the string"), 
        "\n    return text[::-1]\n", 
        ((('',), ''), (('abc',), 'cba'), (('level',), 'level'), (('Alex',), 'xelA'))
    ),
    Family(
        "list_sum", 
        "values", 
        ("Return the sum of values.", "Add every number in the list."), 
        "\n    return sum(values)\n", 
        ((([],), 0), (([1, 2, 3],), 6), (([-2, 5],), 3))
    ),
    # Extra funky ones
    Family(
        "exponent", 
        "x", 
        ("Return exp(x)", "Return the exponential function", "Compute e to the power of x"), 
        "\n    import math\n    return math.exp(x)\n", 
        (((0,), 1.0), ((1.1,), 3.0041660239464334), ((2.08,), 8.004473859664687))
    ),
    Family(
        "factorial", 
        "n", 
        ("Return the factorial of n.", "Compute n factorial"), 
        "\n    import math\n    return math.factorial(n)\n", 
        (((0,), 1), ((1,), 1), ((5,), 120))
    ),
    Family(
        "palindrome", 
        "text", 
        ("Return whether text is a palindrome.", "Check if the string reads the same forwards and backwards."), 
        "\n    return text == text[::-1]\n", 
        ((("",), True), (("radar",), True), (("Hello",), False))
    ),
    Family(
        "celsius_to_fahrenheit", 
        "c", 
        ("Convert Celsius to Fahrenheit", "Transform C temperature to F."), 
        "\n    return c * 9 / 5 + 32\n", 
        (((0,), 32.0), ((100,), 212.0), ((-40,), -40.0))
    ),
    Family(
        "fizzbuzz", 
        "n", 
        ("Return Fizz, Buzz, FizzBuzz, or the number.", "Standard Fizzbuzz logic for a single number."), 
        "\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n", 
        (((3,), 'Fizz'), ((5,), 'Buzz'), ((15,), 'FizzBuzz'), ((7,), "7"))
    ),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}

"""Appends a random 12-letter suffix to a function base (e.g., increment_ajsdyuwnemqx), ensuring an AI model cannot rely on pre-memorized internet code scripts based on standard function names"""
def _name(rng: random.Random, prefix: str) -> str:
    suffix = "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
    return f"{prefix}_{suffix}"

def make_task(family: Family, *, split: str, index: int, seed: int) -> CodingTask:
    rng = random.Random((seed+1) * 1_000_003 + index * 9176 + sum(map(ord, family.name))) # Creates an entirely self-contained pseudo-random number generator object
    entry = _name(rng, family.name) # (e.g., fizzbuzz_xaybzkjqpmnw)
    description = family.descriptions[rng.randrange(len(family.descriptions))] # Randomly selects one of the multiple descriptions from the Family class: For example, for the increment family, it might pick "Return x plus one." for index 0, but "Increase x by one." for index 1.
    prompt = f"# Complete this Python function.\n# {description}\ndef {entry}({family.arguments}):"
    """Constructing the rigid Evaluation Object"""
    return CodingTask(    
        task_id=f"{split}-{family.name}-{index:03d}", # Dataset ID procedure
        family=family.name,
        entry_point=entry,
        prompt=prompt,
        reference_completion=family.body,
        cases=family.cases
    )

def frozen_tasks(split: str, *, per_family: int = 4) -> list[CodingTask]:
    seeds = {'dev': 1701, 'final': 2909, 'rl': 3907, 'confirm': 8123} #  development (used when writing the code or adjusting the prompts); final (oficial benchmark evaluation); reinforcement learning; confirmation splits (verifies so that the model doesnt overfit on dev or final splits)
    if split not in seeds:
        raise ValueError(f"unknown split: {split}")
    tasks = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(per_family):
            tasks.append(make_task(family, split=split, index=index, seed=seeds[split] + family_index * 53))
    return tasks

"""Siulates the pre-trainig phase of the LLM"""
def pretraining_text(index: int, *, seed: int = 42) -> str:
    rng = random.Random(seed * 10_100_019 + index) # Pseudo-random generator - the same index value will generate exactly the same text, enabling GPU training parallelism (the norm in LLMs with a large number of parameters)
    family = FAMILIES[rng.randrange(len(FAMILIES))] # Chooses a random function from the list
    task = make_task(family, split='pretrain', index=index, seed=seed)
    return task.reference_source # Returns prompt + solution concatenation