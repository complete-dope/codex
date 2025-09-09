---
title : Learning the 'Why' in Deep Learning 
layout : post
date : 2025-03-17
---

https://claude.ai/share/82ba8ee9-f673-4f8a-a89a-9fd380bd0e53 

We all know about DL architectures, but do we know how researchers arrived at this !! 

## How to arrive at a deep learning architecture ? How to make connections between layers and be intuitionally correct ? 

The logic behind a deep learning architecture , how to make the gradients flow from different parameters space and still making sense , and the use of residual connections , layer normalisation .. etc 

Starting with vanilla attention network / transformer network 
Its has 6 boundaries / parameter space

1. Token embeddings (vocabulary → vector space)
2. Positional embeddings (position → vector space)
3. Attention projection matrices (Q,K,V)
4. Feed forward network layers
5. Layer normalisation params
6. Output Projection ( for next token prediction )

## How to make information flow and keep these spaces seperate 

Residual connections : Keeps the information flow while keeping the parameter space seperate ( HOW ) ? 

Layer Normalisation : Standardizes output between modules ,allowing each component to 

FFN : point wise transformation, meaning it treats each position independently. In contrast, attention mixes information across positions. 


## How does a single loss functions work for the whole network ?    

Loss function as single 'north star', that guides all the parameters toward a common goal. Gradient tell each parameters how it specifically contributed to the error. 


"Which direction should I change to help the overall goal ?"


## Residual connection 

The residual connections should not have normalisations in them , as there role is to provide a direct path for gradients to flow through the network . 
Adding normalisation in there will affect the gradient that is the flowing through it !!

The whole role of adding / making residual connections is that it helps in passing the grads between 2 different parameter spaces. 


## Relu vs variations 
Hard bounding , makes the grad directly zero ( after passing through the act we are left with a zero values and gradients wont work on that zero values !! [REALLY ? DONT WE HAVE '+' OPERATIONS ??] Nothing saves from the curse of relu as in grad flow it comes as a mutiplication term in the chain rule 

```
z = xW + b
y_hat = sigma(z)
      
( d (y_hat)) / d (z) )

∂y_hat/∂z = {
    0 if z < 0
    1 if z > 0
}

So this term comes as a curse in grad flow !!
```

So, we tend to use its variation like swiglu , gelu ,leaky relu etc 
that are not hardered around negative values ... 

## Batch dimensions and parallel processing 

In pytorch, using in cuda mode
the pytorch assumes all the starting dimensions as batches ( a,b,c,d ) , here a and b are considered as batches and only last 2 dimensions ( c and d ) will be considered in matmul .. rest is batches and we pytorch will consider it as batch operation 


Block Multiplication for parallel processing 

The most common way to get hold of parallel processing is using the block multiplication, this is particularly useful because in this we have simple mat-mul properties and the operations can be performed in parallel


```python
Block size 4x2

A = [1 2]
    [3 4]
    [5 6]
    [7 8]

B = [1 2 3 4]
    [5 6 7 8]


Segments of 2x2x2

A = [A1]  where A1 = [1 2] and A2 = [5 6]
    [A2]          [3 4]        [7 8]

B = [B1 B2] where B1 = [1 2] and B2 = [3 4]
                        [5 6]         [7 8]

C = A×B = [A1×B1 A1×B2]
          [A2×B1 A2×B2]

```


## backward pass and loss functions !

We typically dont need to write our own backward pass in pytorch as that gets handled by the pytorch autograd function which computes a computational graph that handles the backward pass and stuff !

when we call `loss.backward()` it does the internals and we dont have to worry the computational graph building and accumulating and changing gradients .. 

basically the DS looks like :

For every tensor with requires_grad = True, 

```python 
weight : torch.Tensor (W1 x W2)
grad : torch.Tensor (W1 x W2)
```

CAN WRITE A CUSTOM BACKWARD FUNCTION ( COOL ) !! 


How does cross entropy loss gets calculated ? how exactly does any loss function gets calculated ? 

Output vs Expected 

How exactly does CE work in a Next token prediction model ?  

I am passing the input ( B,T ) batches and token, lets say of size 8, 32 ... 
and the expected output ( B,T ) batches and token

How does that even work ? 

So what happens is, at the core what we need is the next correct and that is called generative AI / rather it should be called predictive AI / 
So once we pass the logits to the CE function it converts that to probabilities and then find the most probable one and compares that to ground truth. 

we get the logits > these are raw scores
comes softmax () converts the above to probability distribution 
Now we have the correct value and the model output probability distribution ... 
we take the prob at that position and sum that up using -log(p(x))

that's how we get the CE loss !! 

How does loss.backward() has access to whole computational graph ?

because in model.forward(x) we already passed all !!
 
## Single head and multiple attention head 

We believe that the same positions in the linear layer when passed through a single embedding tries to cover only a single pattern and we need to learn multiple patterns so that we can make chunk , and each chunk has its seperate subspace (but same latent space) in which it can learn its own patterns !! 


Each head is trying to capture something different ( really ? good knows ? similar to in conv each filter tries to learn something new !! do we really motivate the model to come up with this behavior ?? NO !! ) and the model learns the intricacies of the language !!


## ADDING THE REGULARISATION LAYER !!



## CACHING THE KV !! AND WHY NOT Q AND ATTENTION MATRIX 
[MLA](https://planetbanatt.net/articles/mla.html)


We don't cache the Q cause of memory limits !! 
Computational is not tough storing the values and retrieving the values from memory is tough !! 

We only cache the KV, that is efficient and the attention is not the most computation part in transformers we dont need to maximize storing that part ... 


TODO : code the query part and check whether it all store it all  

Adding Q caching increases code complexity for a relatively modest gain. The most expensive operations in inference are typically elsewhere (like the feed-forward networks).

We dont emphasis enough on the fact that KV cache is the most paining point in inference time .. and we need to reduce it as soon as possible !!    





































