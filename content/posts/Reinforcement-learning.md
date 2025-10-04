---
layout : post
date : 2025-08-23
title : Post training methods in LLM using RL 
---

Tags : Reinforcement Learning, RL , Machine learning , deep learning , ML , DL  , Post traning

[PPO](https://www.youtube.com/watch?v=TjHH_--7l8g)  
[RLHF](https://yugeten.github.io/posts/2025/01/ppogrpo/)  

### Maths 
Reinforcement learning, here the agent takes / decides some action to take based on the current state and other variables present at timestep t, and then its takes that action and a reward is followed and weights are updated based on the rewards received by model 

Consider this basic `hello world` example of RL 

State : Any place / position where the agent can be

Action : Up , down , left , right these are the action the agent can take 

Values : The best place we can go from the current state that maximises the return is called Values, we need to find the value function, like there its (Cell Value = Nearby best Value - 1)

<img width="1151" height="770" alt="image" src="https://github.com/user-attachments/assets/05d333ee-ce5d-4944-b1fc-e3d7be8328a2" />

Policy : Given the current state and the neighbouring values , we can create a policy function , a good policy function is `max(up, down, left, right)` and whichever number gives you the best answer that leads to this policy. 

<img width="1658" height="915" alt="image" src="https://github.com/user-attachments/assets/61f413f7-c575-4475-a982-1601829674bf" />

So this is the policy , followung these arrows we can reach the best / optimal position 

So we need a Value function and a Policy function to navigate the landscape and solve the RL problem successfully. 

#### But the catch is with increasing no. of states , finding the Policy and value function is very difficult 

So we use Neural Nets for approximating both the policy and value models 

### VALUE NEURAL NETWORK  

> Value model is deterministic and PNN is approximated  
> The goal is to find the best direction / way to reach to next state given the current state   

<img width="1833" height="804" alt="image" src="https://github.com/user-attachments/assets/e07862e3-c969-4e5b-bb0e-601e1c5196d5" />

Training the Value Neural nets: 
`L_value ( theta ) = (model_pnn - actual_value )^2`

<img width="1780" height="889" alt="image" src="https://github.com/user-attachments/assets/3ef46ff4-66e0-4336-8003-495356dbeddf" />

### POLICY NEURAL NETWORK 

Policy model tells model where to take the next step to go from the current state and increases the probability in the current direction 

<img width="1275" height="800" alt="image" src="https://github.com/user-attachments/assets/d3d59a62-564e-4978-bb96-a77e379ca2a0" />

Training a policy neural net, we are taking scenarios and seeing the gain if the Value network predicted that in the right direction the gain value is 10, means the actual Value is more compared to the predicted value then its means the Value model should increase the value there and the policy model should increase the probabily of going in that direction 

<img width="4993" height="2253" alt="image" src="https://github.com/user-attachments/assets/92b993cb-c896-42f3-b6c3-7c98a6561b21" />

`Loss_policy_theta = (PNN_t(a|s) / PNN_t-1(a|s)) * momentum in that direction`

To avoid very large updates, we clip it so the step is not very erractic and its slowly 

Its same as we have in optimisers, that is, we need to go to a particular direction

That momentum is called advantage (A) / reward in that direction !! 


## Post Training 
Stage 1 : SFT ( Supervised Finetuning )  
Stage 2 : RLHF ( reinforcement learning with human feedback )  

## SFT 
Supervised finetuning, (SFT)
We have the supervised dataset, that is , the training dataset contains both input and defined output to train the model on and the model learns based on the samples provided like classic supervised training     

```
Dataset: 
Input Examples
Output Examples
```

### PPO 
Value and policy model both get rained at the same time


The PPO model looks like this : 
<img width="2526" height="639" alt="image" src="https://github.com/user-attachments/assets/c2f11a47-49ab-4805-a69d-32dd165c5fc4" />


This is the policy model, this is same as the GPT model with an Linear head at last , that output 1d output that tells how good or bad the model is and its just a combination of a linear layer

<img width="1273" height="414" alt="image" src="https://github.com/user-attachments/assets/5c133ec3-1a60-4dd9-8312-7995c9f11609" />

<img width="404" height="485" alt="image" src="https://github.com/user-attachments/assets/346566ee-8411-4828-8395-7858aa13766e" />

The walkhrough for the whole is like this : 
The KL divergence is to ensure the model doesnt drift away too much in its probabilities and starts giving irrelevant outputs that are not at all required to the user !! 

So to regularise the probabiility we find KL div from a trained SFT model .. 

<img width="2526" height="639" alt="image" src="https://github.com/user-attachments/assets/e6acdd52-cd01-43ca-84d3-12bfbda6a70d" />


### GRPO 

In grpo , we ditch the value model and for updating the post training params we sample multiple outputs from the model , then rerank based on the reward model, then find mean , then find the update signal / advantage 

<img width="2526" height="1371" alt="image" src="https://github.com/user-attachments/assets/10a56456-c241-4de4-b41d-ecd1930cdf78" />

[GRPO video](https://www.youtube.com/watch?v=Yi1UCrAsf4o)


# Post Training (RLHF) methods in LLM Training 

Course  : https://learn.deeplearning.ai/courses/post-training-of-llms/lesson/ynmgf/introduction-to-post-training

<img width="2724" height="1410" alt="image" src="https://github.com/user-attachments/assets/1965cd12-f8bc-42c3-81d8-63989bc5f936" />



The output from the base model is not at all coherent, it doesnt output relevant information at all as it doesnt even understands or has seen the chat-template format   

```

=== Base Model (Before SFT) Output ===

Model Input 1:
Give me an 1-sentence introduction of LLM.
Model Output 1:
⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ ⚙ �


Model Input 2:
Calculate 1+1-1
Model Output 2:
⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �


Model Input 3:
What's the difference between thread and process?
Model Output 3:
⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ �

```


SFT cant be ignored in any case that is very important , its the one that tells model what to output , how to output , and in which template to output .. etc

Rest of the optimization comes as an additional improvement step, leading to better models outputs and stuff ! 

DPO : good for identity, multilingual , Instruction following and Safety

Improves model capabilities because of its constrastive nature .. 

DPO dataset curation starts with the one where we start from the sft template and then move ahead and perform computations 


So its 
```
Fine-tuning : SFT

Optimization : DPO

Online RL : PPO and GRPO techniques 
```


