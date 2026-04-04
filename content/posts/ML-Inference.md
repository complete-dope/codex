---
date : 2026-04-04
title : Inference time Optimizatons that we can do in a DL model 

---
## Torch Compile : 
Dev time optimization, take the computation graph , runs the sample (at runtime) sees how mathematical operations are working together then creates an optimized kernel so next requests coming then runs on the optimized kernels. Doesnt support more quantization methods ( works with limited methods )   


## Tensor RT : 
So this works more at a low level, this checks the underlying GPU architecture like is it ampere , or hopper and then optimizes the cuda kernels for that architecture (basically runs multiple passes) and finds the best one and this is different than torch compile cause it sees the SM's , HBM and other relevant things to optimize it for that architecture. 


## Quantization 
[Beginner Video](https://www.youtube.com/watch?v=0VdNflU08yA)  

Weight only : quantizes only weight , the activations remains in the full scale precision   
Activation Only : quantizes activation outputs, the weights are in full scale    
Both : quantizes activation and weights both   

Symmetric : So the range remains uniform and even if some values are missing we dont consider that , main focus is on symmetry.   
Asymmetric : So the range is not uniformly distributed that is some values can be missing or this could be skewed.   

### PTQ  
Post training quantization   
This runs after the training has been done.    
So here we run results through a calibration dataset (as an optional pass) to make sure we are able to calibrate the results    


### QAT 
Quantization aware training   
During training only we add some quantization modules (QuantStub) those stubs collect information about the running mean and averages that we later use in the 



## GPU cores

### Tensor cores 
The bottleneck of ML / DL is matrix multiplication and one way is to divide those to tiles and let each tile do the multiplication plus addition operation and this gives a factor of improvement   
These work inside cuda cores , so the registers memory usage that cuda cores have that all access tensor core also have.    

### Cuda cores
This is where cuda cores work and that's where we run all our computation that we need to do for an cuda core ..  


