---
date : 2026-04-04
title : Inference time Optimizatons that we can do in a DL model 
tags : ['gguf' , 'onnx' , 'tensorrt' , 'compile' , 'quantization' , 'ptq'] 
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

# Quantization + optimization 
So the quantization depends on hardware that not all GPU hardware support all quantizations so like Ampere (A40) doesnt support FP8 in this case and there quantization depends on architecture to architecture 

[trt calibrator](https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-tensorrt/api-reference/docs/python/infer/Int8/EntropyCalibrator2.html)   

So the flow is   
> Pytorch model  > convert to onnx model > then run the trt calibrator > then convert that model to int8 and infer it from that model    


## Where GGUF fits in this all ? 


## Calibration for a PTQ quantization 

Possible method for quantizations that we use are : 

1. Pytorch based Hooks : it is manually adding hooks to the model's forward pass and then seeing what activation values are we getting and then calibrate model based on the values that are received in it   

2. TensorRT framework : so tensorRT has some calibration techniques defined in there official repo we can use those methods to do these parts

3. ONNX runtime calibration : so onnx also has the calibrator that does the same part , calibrate the values and 

4. Model optimizer by nvidia : This is the industry method for doing the calibration extensive support for tensorrt and calibration techniques by researchers ... 

### Practical learnings from the model

Sequence length is something that is failing us from gains the 
That means the image-token sequence going into attention is approximately:
(170 / 2) * (256 / 2) = 85 * 128 = 10880 tokens 

And this is way beyond what the int8-mha sequence sizes can handle and they can handle max sizes of 512 .. 

And this we have in fp8 and A40 doesn’t support fp8 so we need to go to a different gpu architecture for this 
So the MHA silently falls back to FP16 architecture   

Ada GPU : testing this on L40 , this has fp8 support and is considerably faster than int8 ..

We use polygraphy module for inference prototype and as a debugging toolkit  


ONNX graph : Q/DQ graph 

Memory bound vs compute bound : 
So the context is very large in our case and we need to reduce this to make sure our qkv movement is sorted out .. 

So we can’t use flash attention with tensorrt and in tensorrt file the operations are combined so 



