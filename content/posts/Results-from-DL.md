---
title: "Results from DL "
date: 2025-11-17
tags: ["lora", "results", "learnings"]
author: "Mohit Dulani"
ShowToc: true
TocOpen: false
---

[LoRA Article](https://thinkingmachines.ai/blog/lora/)

Small dataset with no augmentation and multiple epochs with lora-rank of 16, overfits the dataset, in such a way that, the exact input tokens only retrieves the exact output, and changing even a single input token leads to corrupt output 

LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, even with ranks as low as 1.
