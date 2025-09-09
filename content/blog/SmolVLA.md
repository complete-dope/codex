---
layout : post 
date : 2025-06-07
title : SmolVLA paper from HG
---

# [SmolVLA]( https://huggingface.co/blog/smolvla )
 
Vision language action paper used for training real world Robots for task 

Input : 
1. Images of surrronding, 
2. task explanation  in text, 
3. state ( senserimotor snapshot at a given time )
Sensorimotor states are projected into a single token using a linear layer to align with the token dimension of the language model.

Ex: 
```
inputs = {
    "image": image_tensor,  # e.g., [3, 224, 224] RGB image from robot's camera
    "instruction": "Put the red cube in the bin",
    "state": np.array([
        0.12, -1.05, 0.44, 0.31, -0.22, 0.08, 1.57,   # joint positions (7)
        0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00,     # joint velocities (7)
        0.85,                                          # gripper open (1.0=open, 0.0=closed)
        0.45, 0.12, 0.33,                              # end-effector position (x, y, z)
        0.0, 0.0, 0.0, 1.0,                            # end-effector orientation (quaternion)
        # ... any other relevant sensor readings
    ])
}

```


Output from Action expert that predict what action it should take next: 
Flow matching is a way to train the action expert so it can generate smooth, realistic action sequences quickly and efficiently.

Instead of predicting actions one step at a time (which is slow), it predicts the whole chunk of actions at once.

During training, you add random noise to real robot action sequences (imagine “shaking up” the correct actions).

The model learns to predict the “correction vector”—the direction and amount needed to turn the noisy, messed-up actions back into the correct ones.

This process teaches the model to create a smooth “vector field” in action space: for any noisy input, it knows how to flow back to a good, realistic robot movement.

Action states:
```
A_t , A_t+1 , ..... , A_t+N
```


The model tends to learn the correction vector , you add noise to the current state of and then it learns to come back to the state. 

Flow Matching

Instead of copying, it learns how to transform any noisy (random) sequence into a realistic one by following the learned vector field.

This is mathematically similar to “denoising” in diffusion models, but flow matching is more direct and efficient for continuous actions.

During training we take a correct action sequence and then add random noise to it .. the loss function is MSE and model trains to come back to the desired position by predicting the correction vector 

All we are trying to do is imitation learning 

### What happened to RL ? Why is that not being used here


Computationally expensive , for task that can be done by imitation and doesnt require an optimised way, ML works perfectly fine in those cases !!  

