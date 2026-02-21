---
title : Scaling a system reliably
date : 2026-02-21
author : Mohit 
---

## Terms 
1. Ephemeral (efimeral) : short lived functions like google-cloud functions , that are running for per request and has a fixed timeout and at that timeout it sends a SIGTERM to close it.

2. Presigned / Signed URL : Is used for upload / download directly to storage (w/o involving backend in this all), its a time based URL that expires with time, the generator of the link should have access and uploader can just upload ... so this is used in Customer support as well , when we upload images of a damaged item, this is used to reduce load to backend and solve this   

## 

