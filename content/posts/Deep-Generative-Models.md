---
title : Deep generative models
date : 2026-02-10
tags : ['generative' , 'diffusion' , 'llm']
---

complex probability that we cant define using stats we use ML in that to help us with it ( learning from data ) 
Spaced repetition  

## Diffusion LLMs

**Curriculum learning** : teach model easy parts in the beginning tough parts at later stages of training. 

**Knowledge Distillation** : We need to use knowledge distillation over steps lets says if we are running diffusion for 1000 steps then we can learn to skip every one step in between to get a better distilled model and this process we can repeat

Prior : fixed belief before seeing data , Chosen by you , Does not depend on input, Same for every example  
`Before seeing anything, I believe values are usually small and centered around zero.`

Posterior : updated belief after observing data, 
True posterior : p(z | x ) : In a perfect world, an ML model would see an input ($x$) and instantly know the exact probability of every possible hidden feature ($z$).

```
Example A: Image Generation

The Input ($x$): A picture of a handwritten digit "4".

The Hidden Variable ($z$): The models internal representation of "the number 4."

The True Posterior: This is the model saying: "Given these specific pixels, I am 99.9% sure the hidden concept is a '4'."
```
