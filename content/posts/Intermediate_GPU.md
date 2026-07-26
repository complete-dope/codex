---
date: 2026-07-26
title: Intermediate GPU learning
description: Learning ins and out of GPU and doing this processing
tags: ['gpu', 'ml', 'cuda', 'rocm', 'amd', 'nvidia' , 'user-space', 'kernel-space']
---

## GPU resources
https://github.com/karpathy/llm.c
https://siboehm.com/articles/22/CUDA-MMM


## Theory of GPU 

### Basic Terms : 
1. Host : CPU device
2. Device : GPU device
3. CUDA : compute unified device architecture, these are used when we need to do write code for nvidia gpus
4. ROCm : Radeon open compute platform, used when we need to write for amd gpus 
5. MSL : Metal shading language to run directly on apple graphics, for macos
6. Logical : a software construct, something that is written in software to make it simpler / easier for people to understand  
7. Physical : its a physical / hardware level definition that we make here means this is present in hardware 

### Terms: 

1. Threads : So smallest execution unit on a GPU is a thread, its role is to complete an atomic instruction on SM
2. Warp : Group of 32 threads that are sequential and in same thread block is called a warp ( they are faster cause they use Shared memory) 
3. blocks : Group / Collection of threads (around 1024) make up a thread block 
4. grid : group of blocks in a gpu makes up a grid ( where blocks are arranged ) 
5. Streaming Multiprocessors : the fundamental building block of an NVIDIA GPU, this is where the operations are executed on 
6. Tiling : A software memory strategy. It is a technique where you break down large datasets into small chunks ("tiles") that fit inside fast, local SRAM (Shared Memory or Registers) to avoid pulling repeatedly from slow VRAM.
7. SIMT (Single Instruction, Multiple Threads) means that multiple threads execute the same instruction at the same time, but each thread operates on its own data.



<img width="850" height="1008" alt="image" src="https://github.com/user-attachments/assets/f9e469b7-d4f3-46d8-ac18-c83c069bf917" />

---
<img width="1576" height="770" alt="image" src="https://github.com/user-attachments/assets/fcdbad31-8527-47b5-833a-49302d5abf4a" />

---
<img width="850" height="262" alt="image" src="https://github.com/user-attachments/assets/cc5114e2-4ab3-415c-bc6f-504c1b1c06a1" />


### Memory layout in GPU 
memory dimension in GPU 

1. HBM : high bandwidth memory ( usually of size 48/ 80gb or more ) also called as VRAM , but slow to fetch (around : 3.35 TB/s)
2. L2 cache : owned by entire GPU chip , sits between HBM and shared memory (~10 to 12 TB/s )
3. L1 cache : sit directly on the physical Streaming Multiprocessor (SM), caches global memory loads/stores for any thread running on that SM. ( ~30 – 33 TB/s )
4. Shared memory : sit directly on the physical Streaming Multiprocessor (SM), allocated to particular thread block
5. Registers : small storage units from where GPU reads data  

### CUDA language
1. `__global__` : means this function / kernel will run on the device
2. `extern` : declare dynamic shared memory inside a kernel function 
3. Atomics make a single memory operation thread-safe in hardware ( like incrementing a variable )
4. Mutex makes an entire block of code thread-safe by allowing only one thread to execute it at a time.


## Kernel in GPU 
kernel is a piece of cuda using which we write instruction / code over a GPU and it executes them
So we write this kernel code in user space , when compiled using nvcc that gets seperated to 2 files , host code and device code , then using MMIO device code gets to the GPU (without interrupting OS) 

Examples : 

`Naive` addition of 2 arrays 
```cpp
__global__ void array_add(float* out, const float* inp1, const float* inp2, const int N ){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // total threads are = N
  if(idx < N){
    out[idx] = inp1[idx] + inp2[idx]; 
  }
}
```


`Naive` sum of arrays 
```cpp
__global__ void array_add(float* out, const float* inp1, const int N ){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  *out = 0.0f;
  // total threads are N
  if(idx < N){
    atomicAdd(&sum, inp1[idx]);
  }
}
```



`Warp` addition of 2 arrays 
group of 32 threads performing SIMT in same thread block using shared memory 
```cpp
__global__ void array_add(float* out, const float* inp1, const float* inp2, const int N ){
  extern __shared__ float shared[]; // declare shared array in shared memory

  namespace cg = cooperative_groups;
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // total threads are = N
  if(idx < N){
    out[idx] = inp1[idx] + inp2[idx]; 
  }
}
```



`Warp + Tiled` addition of 2 arrays 
```cpp
__global__ void array_add(float* out, float* inp1, float* inp2, int N ){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // total threads are = N
  if(idx < N){
    out[idx] = inp1[idx] + inp2[idx]; 
  }
}
```




