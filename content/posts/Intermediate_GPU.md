---
date: 2026-07-26
title: Intermediate GPU learning
description: Learning ins and out of GPU and doing this processing
tags: ['gpu', 'ml', 'cuda', 'rocm', 'amd', 'nvidia' , 'user-space', 'kernel-space']
---

## GPU resources
https://github.com/karpathy/llm.c
https://siboehm.com/articles/22/CUDA-MMM
Tiling1 : https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html
Tiling2 : https://cvw.cac.cornell.edu/cuda-intro/gpu-performance-topics/tiling


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
2. Warp : Group of 32 threads that are sequential and in same thread block is called a warp ( they are faster cause they use Shared memory) .. this is useful if the data needs to be shared among all like in reduce operations.
3. Active mask : So this is used to define which all threads to use in a warp so a mask looks like this `0x00000001` or this `0xFFFFFFFF` 
4. blocks : Group / Collection of threads (around 1024) make up a thread block 
5. grid : group of blocks in a gpu makes up a grid ( where blocks are arranged ) 
6. Streaming Multiprocessors : the fundamental building block of an NVIDIA GPU, this is where the operations are executed on cuda cores
7. Tiling : A software memory strategy. It is a technique where you break down large datasets into small chunks ("tiles") that fit inside fast, local SRAM (Shared Memory or Registers) to avoid pulling repeatedly from slow VRAM. (memory bound ops with high data reuse)
8. SIMT (Single Instruction, Multiple Threads) means that multiple threads execute the same instruction at the same time, but each thread operates on its own data. Here we need a global thread index , loads 
9. SIMD (Single instruction multiple data) : TODO
10. Tile kernel : level of entire tile block, load a whole tile, perform ops on that tile and store back the tile
11. Single thread : There is nothing as single thread in a GPU execution. even if you define `<<<1,1>>>` its basically initiating a full 32 thread hardware warp to execute a single thread. Just that active mask changes in this case 

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
5. `cooperative groups` : easy way to allow and manage groups of threads that can synchronize and communicate with each other
6. `__device__` : this means this is a utility function that will run on one of the kernels of the GPU
7. `__shfl_down_sync` : One thread can read value from another thread that is in same warp
8. `reduce` ops : that takes a list of array of items and combines them into a single value.
9. threadIdx : these are local x,y coordinates inside a thread block ( not global) 
10. blockIdx : this is the thread block idx in a grid
11. blockDim : x, y dimension of a thread block this tells how many threads are present in a thread block 
12. gridIdx : this is a grid of thread block
13. __syncthreads(): 

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
__global__ void array_add(float* sum, const float* inp1, const int N ){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // total threads are N
  if(idx < N){
    atomicAdd(&sum, inp1[idx]);
  }
}
```


Cooperative group based `Warp` addition version array 
group of 32 threads performing SIMT in same thread block using shared memory 
```cpp
__global__ void array_add(float* sum, const float* inp1, const int N ){
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block(); 
  auto warp = cg::tiled_partition<32>(block); // divide the thread block to warps

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float value = inp1[idx];

  // warp computes its own sum
  float warpSum = cg::reduce(warp, value, cg::plus<float>())
  
  // Only one thread per warp updates global memory
  if (warp.thread_rank() == 0)
  {
      atomicAdd(sum, warpSum);
  }
}
```

`Warp` based addition for an array 
Group of 32 threads performing SIMT in same thread block using shared memory 
```cpp

#define FULL_MASK 0xffffffff
__device__ float warpReduceSum(float value){
  for(int offset = 16; offset > 0; offset /= 2){
    value += __shfl_down_sync(FULL_MASK, value, offset); // so the offset value here, offset is a variable that we are looping here .. 
  }
}


__global__ void array_add(float* output, const float* input, const int N ){
  extern __shared__ float shared[]; // declare shared array in shared memory

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float value = input[idx]; // each thread loads one value

  // lane inside the warp 
  int lane_id = threadIdx.x % 32;

  // warp-thread reduction
  value = warpReduceSum(value);

  // only lane 0 writes the warp result
  if(lane == 0){
    output[blockIdx.x] = value;
  }
}
```


`Warp + Tiled` matmul of matrices 
```cpp
__global__ void array_add(float* out, float* inp1, float* inp2, int N ){
  // inp1 : NxN
  // inp2 : NxN
  // out : NxN
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // total threads are = N
  if(idx < N){
    out[idx] = inp1[idx] + inp2[idx]; 
  }
}
```


### Tiling vs Warp 
Warp, its a hardware defined execution concept, executes under SIMT ( single instruction , multiple threads) model. So if in a warp a single thread (thread 0) is doing / completing an instruction then thread 1-31 are physically forced to execute that exact same instruction at the instant.





