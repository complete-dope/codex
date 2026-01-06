---
draft: false
tags: ["kld", "flow matching", "maths for deep learning", "mathematics"]
title: Intuition / Thinking point for AI explained   
date : 2024-06-10
---

[KL-Divergence](https://www.youtube.com/watch?v=tXE23653JrU)
[flow-matching](https://www.youtube.com/watch?v=7cMzfkWFWhI)

Generative models are very good / best in class approximators of a complex probabilistic equation/distribution that we most of the times have no idea of !! 


Images modality : All images in the world are from a very complex distribution of pixels that gives direction , based on the prompts , and are dependent of what the user wants , and as it very complex to model that distribution we rely on NN to predict it , hence we have diffusion models

Text modality : Same as above 

What about Classification models ?   
There also we have true distribution and we need to minimize the output distribution and to minimize that distribution we use cross entropy as its a the term in KLD that is actually dependent on the model params !!

`KLD = entropy(P,Q) - entropy(P)` , deviation between true and unknown distribution !!  

`entropy(P,Q) = cross_entropy(P,Q)`

Cross entropy works like how much suprised you are seeing the output from model   
Consider these examples : 

```python
1. if the predicted probability is 0.1 and the true was 1 so the loss is calculated as : 1 * ln (0.1) = -2.3
2. if the predicted prob is 1 and true was also 1 so suprise is 0
3. if the predicted probability is 1 and true is 0.1, this case will never come (as we dont take that case, we only consider taking the correct probability case always)       
```


So this way we calculate the loss !! and update the parameters / weights of the model ( aka the complex equation estimator )  


Diffusion models they go a step ahead, they start with a gaussian distribution and try to approximate the data distribution


### The maths 

Event : occuring of something is an event , today being a sunny day, avg day , rainy day .. these are all events ! 


Information of an event : the amount of bits required to represent an event , so to minimize this information transfer, the more probable event should have less bits to be transferred and vice versa, that is, p(x) is the prob of event , log(1/p(x)) if p(x) == 1, means this event will definitely happen , so info transfer is 0 , and if p(x) = 0 , then log(1/p(x)) , very high info / bits are needed to be transferred


Entropy : The Avg bits / information required to tell any event to happen from a prob distribution is called Entropy. So sum of (probability of event * information / bits required for the event ) is Entropy of that random variable. 
so `H(X) = - SUM ( p(x) * log(p(x))` 


Cross entropy : So assume we have 2 prob distributions, ( in most of the ML cases , we only know the outcomes / event that occur from an distribution and not the actual underlying distribution ) so we have events and we try to approximate the underlying distribution seeing the events from the original distribution .. so the case becomes we have the dataset that we collected (we should have wholesum data, that should be able to represent every-possible outcome), that is the P(x) and the one are trying to make / estimate that is : Q(x) so the equation becomes as : `- P(x) * log(Q(x))` that is probabiility of getting this event * information bits required to represent this from the other / estimated distribution !!





