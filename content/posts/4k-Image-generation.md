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
* training free solution for getting to 4k resolution images
* 
ResDiT's controlled ablation says native hurts detail. Their finding was that positional embeddings govern layout, while the attention receptive-field scale governs detail fidelity. Going native at 4K means every token attends across a field 16× wider than anything the model saw in training. Their fix for the resulting blur was to reintroduce patch-level local attention at base resolution — which is tiling. You'd be paying a lot to create a problem and then paying more to undo it.



* Tiling works better with sd3.5 also has stablility factors for 4k so its worth a shot ! 

