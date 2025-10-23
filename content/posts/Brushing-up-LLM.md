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
<img width="526" height="346" alt="image" src="https://github.com/user-attachments/assets/b30b87c2-dc3f-4c46-81bc-8100553d264b" />


### Questions to ask ? 
1. How is graphRAG different from RAG ?  
RAG limitations : its good for QA / Summarization tasks but when it comes to multi-hop reasoning , Relational linkages across documents and Indirect relations between documents it fails miserably there  
On the other hand , GraphRAG is good for extracting relations between documents , so retrieval is not just by embedding similarity but also via graph traversal / multi-hop paths to gather context. So the main step in this is the `community detection step` , more better the connection between communities more better the answer and we rely on entity extraction to do that part (if similar entities then they will get connected in the community detection step)   

2. How to implement this from scratch ? **System design** for this retrieval based generation ?   

<img width="830" height="470" alt="image" src="https://github.com/user-attachments/assets/1a47b24f-2bc7-4ee0-bad3-72f8d803e016" />

--- 
# DL Specifics

## Activation and Norms   

Read : https://complete-dope.github.io/codex/posts/activation_normalisation/  

---


## Loss Function 
best video out there to understand Entropy , CE , KLD : https://youtu.be/ErfnhcEV1O8 

Loss functions and likelihood explained : https://complete-dope.github.io/codex/posts/statistics/

Higher weight values means the model has become more sensitive to a small change in the input data i.e. model has now overfitted to the training data 

---

## Optimization Functions

Read : https://complete-dope.github.io/codex/posts/ml-optimisers/

Transformers architecture in forward pass learns what gradient update would have learned in backprop so its a kind of mesa-optimizer way : https://arxiv.org/pdf/2212.07677   

The whole idea is take this 

```bash
y = mx

L  = 1/n  * (y^ - wx) ** 2 

then dl/dw is :

1/n * SUM 2 * (y^ - wx) (-x)

then,

2/n * SUM (y^ * x - w * x^2)

and is w_init = 0 , then
=> 2/n * SUM (y^ * x)
```

And for the other one, 

```bash
h = Q*W.T / (D) *  V

breaking that gives

h = SUM _tokens 0 to i_ Alpha(j) * V

```

<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/8eb7b71f-1fc9-4e43-9c41-047d462c6077" />

---
## Additional / ignored parts

* RMSNorm : works on each dimension independently rather than batch or a layer norm and is fast / easy to use. The role of norm is to just scale down that dimension , not to change its direction or meaning

* Rotary Embedding (RoPE) : This is a nice interesting topic, without positional embedding its hard for model to make sense of what is the word sequence so `I bought a apple watch` and `watch I buy an apple` these 2 are embedded as same only so this clearly makes no sense so first method is to avoid this and add absolute postional embeddings(APE) that is explicitly tell which position them token is at something like `I am token #5` so the same token at different position would mean something else and this was also a flawed approach.

So now we use this RoPE the idea is to encode positional embeddings as rotational vectors using `sin and cosine` such that we encode only the difference between tokens positions and not the Absolute position of the token.   
so the idea is to transform the query and keys vectors in different frequencies to they capture local and global patterns   

> all this intuition is coming from signal processing in electronics where signal at different frequency capture these patterns

so here we have taken a large value ( `N = 10000 ` to capture the local and global features and then distribute d) 

<img width="576" height="350" alt="image" src="https://github.com/user-attachments/assets/0ea7da44-3f26-4f33-b409-e00dfe519993" />

```bash
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves  ( this is done in half like this to make it faster we could have taken as (x1,x2) (x3,x4) but no, for making this operation faster in contiguous memory we are doing this  
    # (x1,x2,x3,x4,x5,x6,x7,x8) ==> (x1,x2,x3,x4) & (x5,x6,x7,x8)
    y1 = x1 * cos + x2 * sin # rotate pairs of dims 
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out
```

<img width="525" height="350" alt="image" src="https://github.com/user-attachments/assets/74af3251-0d51-40bb-a921-f13c35e2be97" />


> When a thing makes no possible sense, there might be computation benefit involved in that !! 

--- 
# Inference time optimisations
Inferencing an LLM is a seperate engineering disciple

* KV-cache : Inference will be done for a single input, that is, one input at a time and only the last token need to attend the keys and values over the rest      
   
* Multi query Attention :   

* Forward pass as a gradient descent operation in it : https://arxiv.org/pdf/2212.07677    

* Chunked Inference (blockwise parallel decoding with masking) : The latest LLM's dont produce autoregressively anymore they do that in chunked manner that is 4-5 tokens or more produced at a time. So here we pass in the same query and now want model to complete it with next tokens and instead of inferencing through all the `16 transformers blocks` one-by-one we do it in a chunked manner that is apply mask to next tokens that the model will produce so that it depend on prev tokens only and even if the internal representation of the token changes rather than being static we are okay with it .. this saves time and makes efficient compute . The forward pass is done for multiple tokens at once (in parallel). No recent model uses this afaik but its interesting way to decrease the inference time  

<img width="560" height="350" alt="image" src="https://github.com/user-attachments/assets/b88dc758-5616-4e8e-8406-0487fa85b6e8" />


--- 


# Basic facts to revise 
We run on batches to do gradient accumulations and then update in a single step so that we dont end up with jittery gradient updates 
* temperature : this is the value that scales down the logits before calculating the final probabilities `logits = logits / temperature`  

So, Temp = 0.99 , this retains the logits value almost as it is   
Temp = 0.50 , this scales down the probability values and makes space for more tokens to be accumated 

<img width="546" height="350" alt="image" src="https://github.com/user-attachments/assets/87cf4d4f-3595-470d-848a-1ea4af598e3f" />


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
<img width="472" height="350" alt="image" src="https://github.com/user-attachments/assets/15a04aab-8db2-414b-8fe1-f6e238bf8e5d" />

<img width="472" height="350" alt="image" src="https://github.com/user-attachments/assets/a4a141f6-e9b4-40e5-a07a-8f8484d3d77e" />



RLHF in production challenges 
To find the model's accuracy we used `LLM as a Judge` , a recurrent pipeline to send data to pipeline and sent to our red-team for HITL  

<img width="640" height="350" alt="image" src="https://github.com/user-attachments/assets/9a08b5f2-789b-4c56-82d2-0740c9d39f4a" />

--- 
## Productionizing system 
Serving a production model and fixing it  
1. Data collection  
2. Data filtering  
3. Feature engineering  
4. Architectural descisions   
5. Model training   
6. Model validation / Eval / Testing   
7. Model packaging and serving
8. Deployments  
9. Live metrics

Challenges in deployment : 
- deploy an optimized TensorRT model for faster inferencing and use  
  
1. Data drift in online training -> can be solved in 2 ways , one is passing it throught a cleaning before training the model on it and other is to use KLD to prevent drift     
2. Spiky load / load burst during campaign : so using elasticity to handle this (horizontally scaling LLM pods), we use queue for buffering and pass the requests in batch mode   

--- 

# SYSTEM DESIGN 
Words to use :  

Hashing : if the search query is too long to search or too long to cache we can hash it and store to reduce the bits stored / transferred

Data ingestion : High Throughput , Streaming , Kafka  
Data parsing : Apache Tika,  Unstructured.io  
Data storage : Object storage ( blob ) S3  
Feature store , Low Latency : repeatedly used Redis (uses an hash-map for O(1) time retrieval )
Model serving speed : Quantization , Pruning , triton inference , kubernetes   
Model serving efficiency : GPU utilization , P99   
Model serving hardware : Inference Optimized GPU's   
Agent Reliability : Stateful , stateless , fault tolerance    
Security : Injection attacks    
MLops tracking : wandb    
monitoring : grafana , elastic-search , kibana (logs)   
Spiky load / sudden load : load balancer , Elasticity ()   
Batching : Batching, it helps to reduce inference speed in models and cost also  

## 1. Design an efficient RAG system   
PS : Design a secure, low-latency RAG system that allows financial analysts to ask complex, natural language questions over 50,000 internal, unstructured PDF documents.

* Data collection : Document Pre-processing & Recursive Chunking - So if the documents contains both text and images and both are relevant , so will use pdf / ocr tools ( like apache tika ) to extract out the text and for images (images in pdf are stored in a binary xobjects ) so the parser can find that out and use an VLM to understand Charts / text documents from it
  
* As its a complex data and requires reasoning we will be using nodes and entities to create a KG and perform a triplet generation this is called Knowledge Graph Augmentation that explains the graph store relationships which are better for multi-hop reasoning

* Dual-Indexing : Using both the graph and the embedding vectors to query in an hierarchical manner. (we can avoid this vectorDB if we keep it in text and do keyword scoring, works on sparsed domain sets) 

* VectorDB use a cloud solution livke Pinecone , Milvus ... so we need to store metadata like citations / pdf-name ... etc so that if a PDF gets updated or removed that smae effect can be shown in the DB also   

* Most important point, how will you evaluate this whole model ?
Creating an evaluation set , basically QA pairs from the ingested context , made by human or an LLM then we can match the outputs using content overlap / LLM as a judge / human in the loop

Post-deployment :  
For internal metrics, we can check citation overlap so we can track what all chunks it retrieved from the chunk metadata and then use a citation retrieval rate and content overall.

We can use HITL to check for accuracy and add guardrails if a confidence-score is low ..   


## 2. Multi-step AI agent orchestration
Design a highly reliable and scalable platform for a multi-step AI Agent that performs a complex, high-value business task, such as automatically generating a financial risk assessment report.  

Step 1 : Clarifying question ?   

1. Noob 1. What is the nature of data and how will it be structured in ? does it contain only text or something else ?
Pro 1. what are the key non-text data types ? Are we analyzing real time market data / high freq logs or static quarterly filings ? Does the data contain PII or proprietary secrets ?

2. What is the expected latency p95 and accuracy and if a tradeoff which should be preffered?

3. How do we handle multi-agent state and rollback ? If agent fails on one step, do we retry, rollback , pass it to human ?
 
 
Step 2 : High Level Components ?   
> Stateful system : Remembers information from one request to the next 
> Stateless system : Request based only on the information provided in a single request. Doesnt need to remember the previous state

LLM (text to sql) -> Data-warehouse -> retrieved data passed to financial LLM -> output  -> retrieved data also passed on to the general purpose LLM   

Agent state store -> redis / DB to save intermediate output
Logging -> pipeline to capture every single step of the agent's execution

All Agents needs an Orchestrator, that kickstarts this service 

Service-1 Data retrieval ( text to sql, sql sanitization , query to data-warehouse ) 
Service-2 Financial LLM (raw to structured output) 
Service-3 report generation 

* Search : 
Text to SQL , sanitize SQL using deny-list to reject any SQL that contain desctructive keyword
Scheme Validation : allowed sql queries / keywords only


* Analyse :  



* Synthesize 


## 3. Design a robust, scalable, and secure pipeline to manage the continuous refinement of a large production LLM using the Reinforcement Learning with Human Feedback (RLHF) process.

Data ingestion : high quality human pre
Model Training : 
Security and Privacy : 

CLARIFYING QUESTIONS : Pick up the single most important thing in the problem statement / most important thing in the whole P.S. 

HIGH LEVEL : 

Step-1 : Clarifying questions for scale and sensitivity ? 
1. Total time / Frequency of the pipeline job ? is it once a week or everyday ? 
2. Do we need to filter out the data as per some rules as dataset would be very important case for this . Are we dealing with some private data ?
3. Initial source of the High quality preference data ? (Training the reward model is the hardest part)
 
Step-2 : High level concepts 
Data collection method : whose role will be to actually collect out the data 
Data filtering process : To filter out the junk and only keep the valuable The quality, the shiny data 
Human annotators / Reward model Training : That would actually do the ranking of whether this is a good output or a bad output 
Policy Model Training and Deployment services : PPO guided by reward model to fine-tune final LLM   
 
Step-3 : Deep-Dive    
1. 
> DON'T DO THIS : For Data collection, we will storing in a MongoDB with the question and answer , so we will use a cron job to query the DB when the load is decreased so that we dont scale up mongodb and increase our cost . We will store that in a jsonl file once collected we can pass that to the filtering pipeline that will look for any forbidden word or match for any dates we can use both a rules based or a SLM to classify it better (THIS IS A POOR CHOICE, ITS SLOW AND INTRODUCES LATENCY)    

> DO THIS : we need a high throughput , real-time streaming architectures , anything real-time needs kafka , producer sends messages to a topic , topic is the named log and consumers read those messages from the topic. so kafka logs it rather than queue , multiple consumers can read a topic from there,  


2. 
> DON'T DO THIS : For Human annotators : The input structure will be <Q, A , R> , R is the reward that they gave and then we can create our policy like `Reward = reward_from_policy - beta(KLD(pie, rho))`
> DO THIS : RLHF structure is wrong needs <Prompt , response A, response B> for policy model to align it to a particular weight 

3.
> DONT DO THIS : For Policy training we can use : transformers trl library to run the pipeline 
> DO THIS : Using a Distributed Training Framework, and running the pipeline in cloud 

## 4. Your company is launching the LLM agent product to a large enterprise client, resulting in a sudden 10x spike in hourly API requests. Design the model serving architecture to handle the sudden load while keeping the P95 latency under 500ms and reducing cloud compute costs by 30% month-over-month (MoM).

Step-1 : Clarifying question ( here the most important thing is the input-output size)

1. What is the average and maximum token length ? 
2. P95 Model latency
3. Is the Model sparse?

<details>
  <summary>Sparsity of model</summary>
  Sparsity of model means to avoid unnecessary computation and it's more of writing efficient kernels that makes fewer multiplies and makes the model faster if kernel supports it. 
</details>

To reduce latency we can use 8-bit or 4-bit integer quantization basically reducing the model size and bandwidth that leads to speeding up token generation 
Dynamic batching, processes one request at a time, system uses queue to group requests and executes them 
Horizontal model spliting : To split the layers across multiple GPU's or even across machines , splitting matmuls like splitting in multiple heads. 
Flash attention for fused kernels 

## 5. Handling load spikes / Designing for elasticity 

Frontend layer : Load balancing and throttling / Rate limiting to prevent a single bad actor from crashing the system 
Queueing (Buffering): After the API  gateway dont ping the server automatically , use a Message queue to prioritise request (if required ) else put them in buffer  

Model serving layer : Auto scaling using horizontal pod autoscaler ( HPA ) , resource reprioritization (if spinning up a resource takes time) , Using dynamic batching 


Using KV cache to cache the key-value tensors for latency optimization  






























