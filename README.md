# About-LLM-
Some implementation about LLM.
## Systems
A triton implementation of FlashAttention-2. Passed all tests in test_attention.py in cs336 assignment2.

![tests](./systems/tests.png)

### Benchmark
Using a script to compare the torch implementation and triton implementation, reached a high acceleration rate in middle and large dataset.

![banchmark](./systems/banchmark.png)
