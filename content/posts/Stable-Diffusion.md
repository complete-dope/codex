---
draft: false
title: Diffusion and flow matching 
date : 2024-06-09
tags: ["FM","flow matching","diffusion model","stable diffusion","DDPM","sampling","VLA","Action tokenizer","colab","code"]
---

The pre-requisites for this blog post is Journey till diffusion model 

## Stable Diffusion model:

Uses cross-attention for allowing conditional modelling ( using text/segmentation map + image to generate image) 

Mode collapse doesnt happen in likelihood based model and SD is a likelihood based model

High frequency details : it means the details / detail-oriented view 

Related work: (previous work in this field) , same vq-vae and vq-gna 


What is the *Inductive bias* of the DM’s inherited by the UNET model ? 

Inductive bias, in machine learning, refers to the set of assumptions a model uses to predict outputs given inputs that it has not encountered before

Diffusion Models (DMs): The inductive bias of diffusion models comes from their underlying process of gradually adding noise to an image and then learning to reverse this process. This approach assumes that:
Image formation can be modeled as a gradual process from noise to structure.
The reverse process (denoising) can be learned and applied step-by-step.
U-Net Architecture: The U-Net, originally designed for biomedical image segmentation, has its own inductive biases:
It assumes that both local and global features are important for the task.
It preserves spatial information through skip connections.
It assumes that features at different scales are relevant for the final output.

When generating an image of a cat, the model can simultaneously capture both the overall shape of the cat (global structure) and fine details like whiskers or fur texture (local features). This is possible due to the U-Net's multi-scale processing combined with the diffusion model's step-by-step refinement.

perceptual loss : extract features from a feature extractor and then calculate the loss based on that low feature values (as passing the model from encoder will remove the high feature values ) 

## Training the autoencoder ( its a universal autoencoder) 

They used perceptual loss along with patch based adversarial loss so the image reconstruction is good !! 

To avoid high variance latent space, we used 2 methods:
1. KL-regularisation between the standard normal and the learned latent .. [ Q-phi ( Z | X ) ] .. similar to vae model 

2. Vector - quantisation ( to limit the generality to a codebook vector) such as in vq-vae and vq-gan papers .. as the output from the encoder is a 2d matrix .. hence its a compressed space already  


Diffusion models are reverse of the markov chain where we get to the get from data distribution to a normal distribution and in DM we go from normal distribution to a data distribution.
 
Reweighted variant of the variational lower bound on p(x), which mirrors denoising score-matching.


What we want in `diffusion model` ? 
Forward diffusion 
q(x_t | x_t-1) = prob of getting x_t from the x_t-1 

We want to find p(x0|xT), but this is difficult to calculate directly.
Instead, we can use Bayes' rule and the Markov property of the diffusion process to break this down into steps:
p(x0:T) = p(xT) ∏T(t=1) p(xt-1|xt)
To get p(x0), we integrate out all the intermediate steps:
p(x0) = ∫x1:T p(x0:T) dx1:T = ∫z p(xT) ∏T(t=1) p(xt-1|xt)


This formula tells us that to generate a clean image (x0), we: 

a) Start with pure noise (xT) 
b) Gradually denoise it step by step (∏T(t=1) p(xt-1|xt)) 
c) Consider all possible paths through this denoising process `(∫z)`

`The model learns to estimate p(xt-1|xt) for each step, which allows it to gradually transform noise into a coherent image.`

# Learning the in's and out's of the diffusion models

It all starts with the real images that people have created , we now want to create more using machine only , so the thought process goes out like :

1. All the images ever made comes from a very complex probability distribution that we are unaware of
2. If there is a way to sample out some example from that distribution, we can generate unlimited images 

So these assumptions are still true but are all unknown, neither we know the underlying distribution nor the sampling method 

```bash
Few learnings: 

* Discrete random variable : so probability of getting a 3 on a dice roll is 1/6

* Continuous random variable :  The chance of hitting exactly 2.00000…8 is zero in an infinite line of possibilities is zero, so here we calculate probability density, The density function 𝑝(𝑥) tells you “how thick the probability is around this point"
So the actual prob. is : P(x belongs to [a,b]) = a_integral_b [p(x)*dx]

* Probability density : probability per unit length aka area under the curve, so P(A <= x <= B) its the area under the curve !  

* Point : In a data field, if data is numerical then this can be [1,2,3,..] if the data is images then its R_d ( 64 x 64 x 3 ) = 12288 dim these many dimensions , where each dimension represent a pixel value. The axis 1 represent value of red channel in (0,0) place ... till axis 12287 ... and the max we can vizualise is till 3 dims

Q) Here this is a discrete random variable right ? the pixel value in an image ?
> Yes, if in 0-255 channel space this would have been discrete but treating them as continous allows us to add gaussing noise, compute grads, use nice maths .. so we convert from 0-255 (discrete space) to 0-1 ( continous space ) -- (1)

We need it continous, so that we can see how the function changes over infinitesimal changes in input and that way only we will be able to see the score / vector field 

```

coming back to (1) , even adding gaussian noise to the real discrete pixels that also would make it continous 


So, we start from unknown distribution of real images then add gaussian noise from 0 mean , I variance over them for a fixed timesteps ( t = 100 ) the image now converts to fully gaussian distribution

Q) But for this gaussian distribution do we know the mean , variance?
> yes, the noise we added was from 0 mean and I variance and after t timestamps we reached there only so we know the mean and variance in this case

The equation for the forward process is : `x_t = [root(alpha_t)] * x_0 + [root(1-alpha_t)] * epsilon`

q(x_t | x_0) : what possible distribution of x_0 would have led to this particular x_t

So instead of relying on all the intermediates steps we use reparameterization trick which is essentially, using closed form solution for gaussian maths and we just sample in one shot this is done for training not in the inference part 

<img width="779" height="474" alt="image" src="https://github.com/user-attachments/assets/5594b3ec-a980-457f-ae82-b69e3fae8e1d" />

### Training

Why do we emphasise on predicting noise value ? 

The field vectors that tell from any point in data field, how to come back to the original distribution that are computed using score vectors and to optimise / learn the field it all comes down to predicting the noise 

log(p(x_t)) :  tells you the direction in the data space where the probability is higher — “which way should we move x_t to make it more like real data.”

<img width="609" height="254" alt="image" src="https://github.com/user-attachments/assets/5bdc1fd8-9ff3-4fed-b99b-f0755fb9fc40" />

```bash
Derivation of the above image :

The equation of gaussian distribution is: 

```

Diffusion is stochastic ,the value of noise (e) also depends on the timestamp   
the forward path is stochastic, each step adds fresh gaussian noise so differentiating it and getting the velocity field makes no sense and therefore we have to rely on probability density , 

Refer to this :  https://colab.research.google.com/drive/1JotCHZTMbh673ndWlbPbRvYKweljOWaq?usp=sharing 

## Flow matching

Its doesnt uses any these above one velocity fields , score fields etc nothing is used .. it simplifies the whole flow 

No noisy forward process in this, we still add noise and make it a gaussian its just that its not 

yeh so here the researchers said to avoid the step wise adding and rather they went on to : `x_t = (1-t) * x_0 + t * E` , where E is the noise vector .. 

`v_t = E - x_0` , so model learn this , here also the model predicts the noise only 

FM is deterministic, the value of noise (e) doesnt depend on timestamp, once its fixed rest all the t=1,2,3... can be sampled out 
The forward path is deterministic, so there we can use velocity field


### Connection to VLA models

this same approach is used in the VLA models to predict the action tokens in robotics . Here we have continous action space in flow matching in tokens we had discrete tokens / space. its predicts velocity in continous space










