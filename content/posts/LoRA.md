---
draft: false 
date: 2025-06-07
title: Low Rank Adaptation
---


# LoRA

This is beneficial only if the rank, r << d, where d is the dimension of the matrix 


## Rank of a matrix

No. of Linearly independent rows/ columns are ranks

A matrix of size 4 x 5 with rank = 2 , can be broken down to ( 4 x 2 ) x ( 2 x 5) reducing the total no. of params from 20 to 18 
And when applied at a large scale for sizes 1024 x 1024 or 2048 x 2048 , and rank = 2 or 4 the size decreases exponentially from (1024 x 1024) to (1024 x 4 x 2) by 128x times

So Rank is used to break down a matrix into small matrixes 


## Adapter Weights
Adapter are additional layers added after the feed-forward in a transformer architecture, and when finetuning the weights are freezed for rest of the model and only this layer weights gets updated but assuming an architecture like BERT where we have more than 24 Transformers blocks these extra adapters params lead to increase in inference time 


## LORA

These first choose the layers that shapes the answer more than other layers then  adds a decomposed matrix of size (d x r) , (r x d) and we fine tune the model freezing original weights and updating the low-rank weights. 

*Target Modules* : q_proj, k_proj, v_proj,  


Now, once the finetuning is do we merge these low-rank matrices with the original weight matrix
so now the weight matrix is : W_q = W_q + alpha * (A_q @ B_q)


This leads to a faster finetuning + no additional weights keeping the original network as it is !!   


