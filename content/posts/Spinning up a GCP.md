---
title : Spinning up a GCP  
layout : post
date : 2025-07-26
---

## Best practices and get a VM in cheap price

Take a GPU GCP , there is a section to select that

then take either V100 or T4 GPU class ( cheap is time and flops are not an issue ) 
+ add persistent additional disk of around 70GB (most important) 

GPU only gcp provides very less SSD storage so this becomes necessary

Then install drivers from nvidia official site and then mount the additional disk and start with placing your project in the mounted device dont touch teh original SSD ... every development and cache goes in the mounted directory not in the other one 

