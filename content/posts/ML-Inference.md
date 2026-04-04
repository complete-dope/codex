---
date : 2026-04-04
title : Inference time Optimizatons that we can do in a DL model 

---
## Torch Compile : 
Dev time optimization, take the computation graph , runs the sample (at runtime) sees how mathematical operations are working together then creates an optimized kernel so next requests coming then runs on the optimized kernels. Doesnt support more quantization methods ( works with limited methods )   
`torch.compile(backend='')` : this uses  
So there are many possible backends : 

1. `torch-tensorrt`
2. 


## Tensor RT : 
So this works more at a low level, this checks the underlying GPU architecture like is it ampere , or hopper and then optimizes the cuda kernels for that architecture (basically runs multiple passes) and finds the best one and this is different than torch compile cause it sees the SM's , HBM and other relevant things to optimize it for that architecture. 
This is one of the fastest method to increase the code speed 
So we need to have a TRT-lowering implementation of kernel as well else it works same as the eager implementation that we earlier had 

Q) Why is that few ops can be used with tensorrt and others cant work with it ? 
A) Cause some ops are not yet covered / or are convered in some other form and as we this acts as a seperate runtime env / and usually acts as a convertor from one runtime to another, so that is made compatible by community or model makers and someone has to do this low level work. It requires all ops to be declared in tensorRT also and this is not an optimized approach as it breaks on unsupported ops    
Leads to a lower vram as less intermediate outputs   

## TensorRT subgraphs 
Subgraphs goes a params in tensorRT that basically fuses ops together and create a small acceleration engine for those ops 
and is we keep min-block-size in subgraph as 1 , then for each op it will create its own small acceleration engine

Potential method around :
1. Convert that to onnx
2. then convert from onnx to optimized engine that is in trt
3. use that trt to do the computations then
4. So the fused kernels are merges ops and lead to a lower peak Vram   


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



# Nvidia System Profiler 
`nsys` : nvidia system profiler 
Tells you exactly the bottleneck of your model , whether its memory / compute or some other thing and is extremely useful 
NVTX : Nvidia Tool Extension SDK
Here in events view we can find which process is taking most of the time and which is taking least amount of time .. 
Before optimizing anything in a new architecture / model always first profile it 



Q) How to know if you are using tensor cores or cuda cores ? 
A) 


# Profiling vs benchmarking 


Q) 

