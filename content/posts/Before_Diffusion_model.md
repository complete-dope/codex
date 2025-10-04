---
draft: false
title: Journey till diffusion model  
date: 2024-03-03
---

Diffusion model's were not built in a day, so we will learn the underlying concepts that helped us reach to diffusion model.


![Loose ideas](https://docs.google.com/document/d/12zcDr7z5qjcMyuKFnKR6C_m7NMAO0Ki43--IpEdJ3FM/edit?usp=sharing) 


#### Autoencoders
These are the most basic type, its used for reconstruction of data, for example, if you are given an image, you can train an autoencoder to reconstruct the image. 
These can be used for dimensionality reduction, feature extraction and data compression. 

#### Variational Autoencoders
They are first in the category to produce new data points from the existing dataset by learning the latent space of the data. 
We produce 2 vectors from the encoder namely, mean and log variance and we sample a vector from a gaussian distribution with mean and variance as parameters.
This latent vector is then given to the decoder to reconstruct the image of the same size as the original image.

Here the marginal likelihood (evidence) of the data is not tractable, so we use a lower bound on the marginal likelihood, which is called the variational lower bound. (ELBO)

Also we use reparameterization trick to sample the latent vector.

The objective is to maximise the marginal likelihood of the data !! 

This is a maths heavy .. refer to the doc and the stats section for more details.

#### Vector-Quantized Variational Autoencoders  

Here also we have encoder that produces a matrix of size, lets say 4 x 4 x d, and a code-book of n x d dimensions ( which is initialised at the start) 
We find the closest vector in the code-book for each vector in the matrix and we update our encoder output to be the index of the closest vector in the code-book.
That new matrix is then passed on to the decoder to reconstruct the image.

Then in the backprop we train enc , dec,  along with the code-book.

In inference, 
randomly sample the indices from the code-book , convert that to the actual inplace values of the codebook and pass it to the decoder to reconstruct the image.
So we need to keep a codebook to get the embeddings from.. 

The loss function used is quite interesting here, the loss function is the real maths used in the paper, 

TODO: A detailed discussion on the loss function used in vqvae, has 3 loss terms, 


#### Vector-Quantized - Generative Adversarial Networks

Similar to the vq-vae model , here we use additional discriminator to discriminate between the original image and the reconstructed image.
Minor changes to the loss function and the training procedure.

In inferencing, the next token prediction is done using a next token prediction (and whose better than transformers on that) models like transformers .. just start with a blank token and keep predicting the next token till the end of the sequence.

We need to keep a codebook to get the embeddings from.. 

