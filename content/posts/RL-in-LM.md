---
date: 23-11-2025
draft : false
title : Reinforcement learning in Language Modelling 
---

Alignment / RLHF part of the models , and how to do those !

Pretrained model to getting to Instruct GPT model the whole flow of going from here to there 
 
Enabling better, tighter controls over LM output 

Post-training : 

SFT : if you want to imitate expert demostration you better have expert demonstration of what that looks like , once we have the data how to adapt to it ?   
https://youtu.be/Dfu7vC9jo4w?t=337 


All policies tries to find / estimate the Advantage value ..
PPO does that by telling how better this action is compared to the average .. Advantage  = ( R - Baseline ) 

and to find that baseline we have another LM called as value network and this is a critic model unless we would have to go out and calculates how better the action at timestamp t is by going ahead in future and taking all possible steps that was possible ( if 10 more tokens or in total 10 tokens to generate VS ** 10  that is close to 6 x 10**51 , this is an impossible no. of calculate , use LLM to generate over those and then find one grad update step) so here policy model comes handy .. we need to use something called as monte carlo approximation 


formally, true value V of a state is the expectation of future rewards (aka average of future rewards )

this is the expectation of the future reward , so we use this value model, that estimates the expected future rewards 

so how do we train out value model to converge to this value ? use a loss function over value model that is MSE of (predicted  -  E (g_t | s_t))

`g_t` = is the expected reward over all the future values of the tarjectory that are starting from this path ( so the path till now is s_t), in this case    

but the only reason to use the value model was to not calculate the impossible mathematical path value the question comes down to how to to find `g_t`  … so one way is to find all the path ( that we calculated above that is just impossible to do ) , so we use MCA ( monte carlo approximation), that says take just one path that the model outputted and assume that is the average value


Value model is a live commentator, that says till now the model score is 0.5, now its 0.6, now its 0.1, now its 0.05 .. like that as the state updates its value also updates … 

and we do that for a single sample per input token , this is a tradeoff and this introduces high variance (single example can lead to low or high value) and low bias  

PPO Models : reward model , policy model , value model 

reward model grades the answer, 
the value model is for the baseline reward model, to tell ohh this is a easy question and model gets this right always so baseline reward as decided by the policy model here is then 10 , so if the model gets a low score from reward model of 5, then its heavily penalised .. 


Pure Monte Carlo, punishes the results equally, but that is not the case we need to find the token that did the mistake of taking everything down that is found using the GAE ( in short , this gives a tokenwise score ) 


GAE works like : in a complex manner , but the intuition is this only 


So this is all very complex to do in real life , a simpler approach is used in GRPO, 


When to use SFT	 vs when to use RL ? 

If the dataset is defined comes with the supervised pair of <input, output> then you should choose a SFT to train your model 

but if we dont have labels and we depend on end rewards to train a model then we do RL

 — 

— training a small grpo policy
The model quickly learns to get more output from the tagging part compared to the correct output part ! , this is similar to how reward hacking works , where model learns to output more no. of tokens for getting more reward 	, so the value model then learns from this reward model and  

## RL setup and working 
So RL works like this : first you need to have questions, those questions are passed to the policy model that you are trying to optimize and the output is the generated response, that response is then sent to human, that gives a process / step wise reward to it (aka process rewarding) , then once this dataset is collected, we pass that to the reward model and train the model on these sequences ( this is how the reward model learns things).

For general chat, we only get 0/1 output from the human which is better or not 
For reasoning, we do process rewarding , verifiers  

So till now the reward model is trained using human annotated dataset


Now we have value model, that is the critic model that tells how good current generation is , this is like a live commentator, and we penalise this with the final reward ( that is for deterministic problems like maths, physics etc we have it in  0 / 1 else for chat question like :’write a poem on rain’, we use reward model that was trained on prev step to output this reward) and the loss function is just an MSE of (final reward - value model tokenwise prediction) , where final reward can come from either deterministic way or from reward model 
but is the value model here really learning to give token wise confidence ? we can use a combination of the deterministic reward + stepwise reward and check it then create the loss function accordingly. But this is also not training that efficiently rather we can use the GRPO and use that ..  

So now we have a trained value model also 

Now for policy model, we need to have an advantage value, that is how much marginalised reward for this current token prediction to take ? like how much should I weight this current prediction that I got from the policy , if this was an easy prediction and I got it right, then nothing much to update, vice versa this will be a nice step to take in this direction 

This is done on-policy, that is generations are taken from policy model only and updated over that 

—
Dataset for same : Birchlabs/openai-prm800k-stepwise-critic , this is used for training reward model


— 
PPO , GRPO : these are techniques to find out that advantage value , and as we saw above the Value model is not learning anything much new , so grpo goes out and  removes that 


DPO : that is preference optimization and the loss function is more like  , its loss function is more like , Increases it over a correct prediction , substract from the wrong prediction .. 

—
MOTIVATION BEHIND THIS RL : 
Thinking models are one way of improving models as we now have step wise critics for the thinking process and this helps in finding where the model went wrong (as the reward function also looks at it step wise, similar to how we find a solution is wrong.. going step by step) and how can we improve it using a better trained reward model.. so this is a reason to bring thinking models into action 
 
