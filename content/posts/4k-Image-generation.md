---
title : Using Diffusion model to generate 4k images
date : 2026-08-19
description : Training a diffusion model to generate 4k images   
---

A lot comes when you are training a model to generate 4k images:  
* Translation from the positional embedding 
* 


Tiling is the only way here : 
1. Cause sd3 training was whole on images that are under 2MP

So tiling requires an conditional full size image as well to see the surrounding context to make correct edits there


Position embedding limitation : 
SD3 : fixed sincos
SD3.5 : Uses RoPE

SD3.5 vs SD3 
* 4096px → 256 patches (), under 384. No ValueError. Native 4K is architecturally available on 3.5-Medium in a way it never was on 3-Medium.
* Same VAE latent channels that are 16 (f16) 
* SD3.5 tops around 2MP
* Tiling here works properly comapared to what we had in SD3 
* Cause  : Better prior (better than sd3), Random-crop position pretraining (DiT tolerate seeing a sub-region of the original positional grid, but you still need a global image here) 
   Nati

VAE channels 
* Also a major factor in reconstruction and getting lost fidelity 
* latent channels that are 16 (f16)



ResDIT paper : (https://arxiv.org/abs/2512.01426)
- Training free solution for getting to 4k resolution images
- Currently in industry we have 4 methods only to generate a 4k image
   * Full training / finetuning
   * 2 stage paradigm, generate a base resolution image then use it to guide high resolution synthesis. (they treat high res generation as super resolution task, relying on external guidance)
   * Tiling based method

- Attention is the only mechanism that enables token interaction in DiT, it has 2 crucial aspects :
   * Positional embedding
   * Attention range field
     
- Their finding was that positional embeddings govern layout, while the attention receptive-field scale governs detail fidelity.
- Going native at 4K means every token attends across a field 16× wider than anything the model saw in training. Their fix for the resulting blur was to reintroduce patch-level local attention at base resolution — which is tiling. 
- Overlapping parts of an image are averaged (using gaussian here )  
<img width="633" height="340" alt="image" src="https://github.com/user-attachments/assets/d4b15abd-86d3-4932-9398-806c92dd142c" />

----

* Global attention over a highresolution feature map often causes blurred textures and loss of fine details because the model is forced far beyond the spatial scale it was trained on, to overcome we use partitioning and splicing techniques like minimum overlap partitioning
* 
* 
* Tiling works better with sd3.5 also has stablility factors for 4k so its worth a shot ! 


Removing VAE completely (https://studio.aifilms.ai/blog/l2p-latent-pixel-generation)
paper : https://arxiv.org/pdf/2605.12013 



SD3 : 
F = 8 , downscaling 
C = 16, VAE channels are 16 
The compression ratio essentially becomes something as : H x W x 3 to H/F x W/F x C so that is : `F x F x 3 / C` so this tells up how much compression is done from pixel space to latent space compression 

Other papers that are doing same : 
* SEGA (https://arxiv.org/pdf/2605.22668)
* PixelRush (https://arxiv.org/pdf/2602.12769)
  
