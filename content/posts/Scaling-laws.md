---
date : 2026-06-17
title : Scaling a transformer model  
---

Following this : https://jax-ml.github.io/scaling-book/inference/#the-basics-of-transformer-inference

## Beam search 
Speculative decoding used at inference time by expanding the most probable nodes for 3 to 5 iterations and chooses the best iteration out 
so here we calculate the sum of log probabilities rather than taking just the normal probabilities 

