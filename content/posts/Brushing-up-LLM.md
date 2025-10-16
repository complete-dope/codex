---
draft : false
author : Mohit
title : Brushing up LLM for Interview
description : End to end notes for revision for interview
date : 2025-10-13
tags: ['interview', 'preparation']
---


# Embeddings 

## Vector Embeddings 

https://www.pinecone.io/learn/what-is-similarity-search/ 

Searching over structured data that is easy we can use Data structures for it like Binary tress / arrays (sorted order) and also things like hashset. This was done in internet 2.0 , sql , mysql , mongodb these leveraged it so well

Now for unstructured data we need something that represents more deeper concept / representation of the data 

Using sentence-transformers (and models like Word2Vec , BERT model)
* So in the bert model we train it using the `[CLS]` token / prefix, we take the trained model and then extract this token embedding.
* And in word2vec model, we use cbow and skip-gram that depend on the proximity of similar words


## Vector Search 

Terminologies : IndexPQ ( product quantizers ) , IndexIVFPQ ( Inverted File with Product Quantization ) 

Given a query vector and multiple value vectors, we need to search a relevant query that is called as a `nearest neighbors search`

Popular methods to use these are: 
1. `kNN` ( k Nearest Neighbors ) : To find k nearest ones we need to get check similarity with all the vectors so that is ineffient its takes `O(n)` time  
2. `aNN` ( approximately nearest neighbors ) : To avoid this exhaustive search we use this ANN , the point here is we are ready to sacrifice on some of the nearest ones to get the speed at scale. This is done using `Index structure`

## ANN ( Approximately Nearest Neighbors ) 
https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/ 


### Compression
So lets say we have 50k images each embedded in 1024 dims and we divided it to 8 x 50k x 128 vectors so it all becomes 
<img width="567" height="270" alt="image" src="https://github.com/user-attachments/assets/db49f1f8-0a65-49d5-b521-e29b6151b0be" />

<details>
  <summary>K Means Clustering </summary>
  
  So in short , first we randomly initialise few centroids and then assign each vector its nearest centroid point then take the mean of assigned vector and change the centroid     vector. This is done repeatedly till they stop moving much 

  Q) What if one centroid has more vector assigned to it ? 
  A) We can keep a limit those anyways, once a vector is assigned a centroid and total assigned values to that centroid is less than (possible_vectors / no. of centroids) then     we will take it else discard it   
</details>

Then we replace those vectors assigned to there custer vector so we reduced from 50k to 256 and each subvector ( 128 size ) is assigned a cluster id from (0-255) and we have 8 subvectors created from a single vector ( 1024 / 128 ) so now a vector of size 1024 is represented in size 8  
 

### Things used in current VectorDB for RAG
#### Inverted File Index ( IVF )  
https://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/

In this we first cluster all the vector using k-mean-clustering into k clusters , so each cluster has a centroid , so at query time we match with all the centroids and take few clusters ( lets say - 10 ) and search from vectors within those 

Goal: Reduce the number of vectors to compare → faster search.
Problem : Still the vectors inside are high dimensional so memory explodes 

#### Product Quantizer 
Then for each vector get `residual (r = x-c)` out from its centroid so everything is now centered around zero ( for all the vectors ) then create subvectors and codebook as done in the compression stage for ANN use it to create a `global codebook` so each vector inside the cluster is now represented in the codebook codes and its a common dictionary made for all the codes this part helps to reduce `memory footprints`      


so once a cluster is selected, we calculate the residual value from it, then create a codebook id for that particular query's residual vector, so this way converted the vector to codebook.

A similar process is done for all the datapoints also that is we convert them to code book and then 8bit codebook search is done in that cluster compared to 32 bit search so that reduces memory time.  


## GraphRAG  
Implemented this paper :  https://arxiv.org/pdf/2404.16130
<img width="826" height="606" alt="image" src="https://github.com/user-attachments/assets/b30b87c2-dc3f-4c46-81bc-8100553d264b" />


### Questions to ask ? 
1. How is graphRAG different from RAG ?  
RAG limitations : its good for QA / Summarization tasks but when it comes to multi-hop reasoning , Relational linkages across documents and Indirect relations between documents it fails miserably there  
On the other hand , GraphRAG is good for extracting relations between documents , so retrieval is not just by embedding similarity but also via graph traversal / multi-hop paths to gather context. So the main step in this is the `community detection step` , more better the connection between communities more better the answer and we rely on entity extraction to do that part (if similar entities then they will get connected in the community detection step)   

2. How to implement this from scratch ? **System design** for this retrieval based generation ?   

<img width="830" height="470" alt="image" src="https://github.com/user-attachments/assets/1a47b24f-2bc7-4ee0-bad3-72f8d803e016" />




## Loss Function 
best video out there to understand Entropy , CE , KLD : https://youtu.be/ErfnhcEV1O8 

Loss functions and likelihood explained : https://complete-dope.github.io/codex/posts/statistics/


## Optimization Functions


# Facts 
We run on batches to do gradient accumulations and then update in a single step so that we dont end up with jittery gradient updates 


## Post training 

### SFT 
https://huggingface.co/docs/trl/en/sft_trainer 
SFT is supervised finetuning , we do to adapt a language model to a target dataset aka make model aware on our dataset, here the model learns about chat-template,  

* SFT dataset creation 
We create QA pairs where input is question and output is answer. This is passed on to model as 
We usually train using the `sft` library by hugging face, and we compute loss on the assistant output and compute loss on it 

<img width="1988" height="350" alt="image" src="https://github.com/user-attachments/assets/7f799aca-72d7-4b5d-9558-839bb1bc9352" />

For daily chat application, we dont want the model to calculate loss on what the user would have sad next in there input so we mask that out, (similar to what we have done above we do the same for the user's questions .. remove labels from those, attention will still be calculated , loss also will be taken out but not added to the loss value)

* SFT Implementation
Using peft (parameter efficient fine tuning like LoRA) that helps in training the adapters 

<details>
  <summary>LoRA adapter</summary>
  Low Rank adaptation, so authors found that finetuned models have low rank weight matrices so they hypothesised that there update matrices will also be low rank and based on that they experimented with this low rank weight update matrix. so we will freeze the current weight matrix , create 2 random matrices of size `(Row x rank) (rank x col)` such that while finetuning we do : `y = W_q * X + alpha(A(B(x)))` so the W_q remains fix we update only the delta matrix and these are called low rank adapters, alpha is the scale value , rank is the low rank value , rank is the hyperparam , more the value of r more expressive
</details>

<details>
  <summary>QLora</summary>
  Quantized lora, so not all weights of the LLM should be quantized you have to leave some weights in original dtype only like layernorm , normalizations layers , bias terms and output logits / final projection layers and we quantize these embedding layers, MLP , Embedding layers
</details>

<details>
  <summary>Loss Value</summary>
  The scalar loss value in itself means nothing, loss came out to be 100 or 1000 that just means that model is performing worse and it has no relevance in the weight updation using any optimizer. 
</details>


Instruction tuning : 
Teaches a base language model to follow user instructions and engage in conversations (wasnt this the purpose of post-training ?) requires chat template ( role definition , special tokens ) and a conversational dataset 

* SFT in Production challenges 

* SFT metrics
LLM based evaluation , Accuracy, BLEU(n-gram overlap matching between model output and reference), Perplexity , preference alignment 

--- 
### RLHF
So its a combination of SFT , then Reward Modelling and then using policy to improve the SFT model . So its a 3 step process 

Reinforcement learning with human feedback, here we train a reward model ( who's role is to output / rank the answer) , the dataset for this training is created by humans preference judgements then we train a reward model to predict the score then freeze that model and     

RLHF is used to make text `good` and as `good` itself is hard to define and is subjective and context dependent and to compensate for the shortcomings of the loss itself people define metrics as `BLEU` or `ROUGE`  (Bilingual Evaluation Understudy)

But BLEU is also rule based and doesnt work well so we use human feedback as a measure of performance and use it as a loss to optimize the model. 

Metrics used : 

RLHF dataset creation : 
So we train a reward model that is the same architecture with a linear layer at end, the backbone is freezed and we just train the output that too basde on the


<details>
  <summary>Terms in RL</summary>
  * Rejection sampling : whatever output we got from the RM that is rejected out from prob distribution based on score   
  * Empirical calculation : the one that is done by doing actual experiments 
</details>

RLHF Implementation



RLHF in production challenges 


<img width="1340" height="772" alt="image" src="https://github.com/user-attachments/assets/9a08b5f2-789b-4c56-82d2-0740c9d39f4a" />













