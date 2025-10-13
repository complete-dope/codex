---
draft : false
author : Mohit
title : Brushing up LLM for Interview
description : End to end notes for revision for interview
date : 2025-10-13
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
 
   


