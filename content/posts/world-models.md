---
title: World Models
date: 2026-05-3
author: Mohit
---

# World models
Its a framework ( emphasised by yann le cun ) that works autoregressively (that is we pass in last x frame and model predicts x_t+1 frame) and 

World model usually cares about dynamics, causality, and actions

so a world models learns next state must we consistent with the world rules ( a ball in air must come down also ) 
Action conditioning in the predictor model 

<img width="1292" height="686" alt="image" src="https://github.com/user-attachments/assets/743ec589-eb15-4b19-b31d-fa92c63f66f8" />
