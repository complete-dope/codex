---
draft: false
title: Stable Diffusion model 
date : 2024-06-09
---

The pre-requisites for this blog post is Journey till diffusion model 

##### Stable Diffusion model:

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


##### Training the autoencoder ( its a universal autoencoder) 

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


