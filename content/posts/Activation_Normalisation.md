---
draft: false
title: "Activation and Norms"
tags: ['layernorm' , 'norm' , 'activation']
date: 2024-09-08
---

#  Activation / Non-linear function

A non-linear function is one in which cannot be expressed from  `y = mx + c` 

lets suppose a function with y = x * W1 * W2 .. Wl

Linearity with respect to x, this means rest of the variable are constant and only x is changing , so linearity with respect to `x` is linear    
Linearity with respect to Wi, this means rest of the variable (`x here`) is constant and only weights are changing , so linearity with respect to `W` is non-linear   

## Tanh 

<img width="646" height="400" alt="image" src="https://github.com/user-attachments/assets/116ad60a-a021-4ec6-9e53-adcddb294a3b" />
The formula goes as : `e^x - e^(-x) / e^x + e^(-x)` and the derivate is : `1 - tanh^2`

So activations are just squishing function, that means, any amount of input that you will pass will lead out a value between `-1 to 1`  

This same thing also brings in non-linearity , rather than it being a hyperplane in high-dimension we need our functions to learn a complex dependent distribution   

## Sigmoid

<img width="160" height="183" alt="image" src="https://github.com/user-attachments/assets/0d171d8f-c2e5-423a-93d5-1da081e6477d" />
This does it between 0 to 1 so no negative values  
This gives a probability range : `1 / (1+e^(-x))`  

## Softmax
This maps it from 0 to 1 probability range, and simply is shown as `e^x / SUM(e^x)`   
<img width="306" height="165" alt="image" src="https://github.com/user-attachments/assets/186eaad7-4a52-441e-9965-128e746e6b99" />


## ReLU
ReLU ( rectified linear unit ) the names itself says that its linear unit but the non linearity comes from the kink at `x = 0`  

Its like linear lego blocks added and given non linearity (like for predicting x^2 distribution)   
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/96bdd3e0-f75f-481e-bc9a-02e934d2c122" />

But if you stack many small, straight LEGO blocks at slight angles, you can build a perfect-looking circle or curve.

## Problem with existing activation functions and how ReLU solves it  
So biggest problem is vanishing grads for actvation function and they quickly get vanished,   
`0.8 * 0.8 * 0.8 *0.8 * 0.8 => 0.3276 ( multiple's of these would lead to vanishing grads )`   

So even the scaling wont help as the problem is with the gradient calculation itself, so relu seems a promising approach   

![Computation Graph](https://www.researchgate.net/profile/Yuqing-Chen-3/publication/344260274/figure/fig3/AS:936580785139716@1600309668673/A-a-neural-network-and-b-its-computational-graph-The-c-forward-and-backward.png)

## GELU ( Gaussian error linear units)
This uses a CDF of standard normal CDF that tells what is the output of getting this value multiplied by the input x so the formula is `x * PHI(x)` , where PHI is the probability distribution of this function
CDF is the cumulative distribution function that tells P( X<= x)  so for a normal distribution , P(X <= 0) is 0.5 , P(X<=1) is 0.84 ... like this !  


# Loss function and calculation 

In the backward pass,   

Derivate: quantifies how sensitive a function is with respect to the change in its the input 

[Loss.backward()](https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944) -> computes dloss/dx for all the parameters x , which has requires_grad in the model 

dL / dWs -> dL / d(wx+b) * d(wx+b) /        dx  
(Ws : all the parameters in the model) -> and is done using the chain rule  


## Revision / paper reading 

Paper : Chameleon paper by meta ( https://arxiv.org/pdf/2405.09818 ) 



### Forward pass Gradient explosion / vanishing
If the weights at backward pass of step t-1 rises , then the weights at forward pass of t step also rises 


### Backward pass gradient explosion / vanishign  
Weights updation happens in the backward pass



### Gradient free meta learning 
Never computes or propagates gradients.
Treats the entire weight set as a black‑box parameter vector and optimizes it with non‑gradient methods (e.g. evolutionary strategies, population‑based search).


Think of your entire set of 4‑bit quantized weights 𝑊 as a single “individual” in a population.
Each individual is just one candidate solution—a full weight assignment for the network.

You start with 𝑁 random individuals (e.g. 𝑁 = 100 ) 
and anyone of this can be a potential case for our case , we randomly start and do just a forward pass calculate loss value 

Rank from lowest loss to highest and pick up top 2 candidates (elitism) ..  
and then in loop = 2, choose random 98 more, total 2+98 = 100 more 

and then we have these possibilities to do: 

#### Mutuation : 

Introduce Small Random Changes
For each new child, randomly pick a small percentage of weight bits (e.g. 1–2 %) and flip them.
This injects fresh diversity and helps explore new regions of weight‑space.
Ensure Quantization Consistency
Since weights are 4‑bit log values, any mutation simply toggles one of those bits—no out‑of‑range values.

#### Quantization :
Nice post on quantization : https://www.maartengrootendorst.com/blog/quantization/

#### Crossover : 

Pair Up Parents
Randomly choose two parents (with or without weighting by fitness) for each crossover event.
Mix Their “Genomes”
For each weight position (4‑bit log value), flip a (virtual) coin: choose the bit from parent A or parent B.
Produce one or two children from each pair.

Reduces the loss over and over , no backward pass 


## Int vs Float 

So this is something interesting, integers are actual no. that are store in the memory , they reduce computational time , and have fixed / accurate value. 

A int32 can hold at max 2^31 - 1
This is used for accuracy and that means its used for list indexing .. 


Float on the other hand is just a `representation` of a no. 
A float32 can hold max of 2^127 and this is how they do it: 
Its its within the bits limit (2^34), its stored as it is , else if we increase this to more then its gets hashed and stored using this

```
sign 
exponent 
mantissa
```

Assume it like a hash function, computer stores the hashed value in float32 .. 
so before any operation performed on float values we need to decompose it to the actual value ( to its computational-value which can take more space for the moment ) , once calculation is done , then it again gives answer back in float so its this .. and in this to and fro conversions they are prone to errors in the accuracy  


# Normalization layers (Batchnorm , layernorm , RMSnorm , l1 norm , l2 norm)

This is applied before passing to the activation so that the inputs to activations are all centered around and we get the best benefits from it   

Deep Learning book : https://aikosh.indiaai.gov.in/static/Deep+Learning+Ian+Goodfellow.pdf 
read section : `8.7.1`

* Normalization layer :  

These techniques are used to speed up the training part / increase model training   
The reason to bring these in action is because, each time the layer (l-1)th increases the layer l also has to change the weights to actually converge before the whole model can reach convergence state and these lead to in internal covariate shift .. so to avoid this delay in convergence we apply these batchnorms for inputs where batches make more sense ( like image related stuff ) , other is , language input where the batches dont make much sense so we go with layernorm ( requires 2 passes through data) , but they also have some overhead to over come these overhead we use rmsnorm (norm that is done in a single pass ) 

These normalizations are used to reparameterize the weights, the problem is the update of weights of one layer depend so much on the other layers that we simply just cant change one without thinking about other one and this makes using dfferent adaptable learning rate for each layer as an impossible problem to solve 

So for this we use normalization ,




















