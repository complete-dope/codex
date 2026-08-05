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
Tiling video : http://youtube.com/watch?v=ccHyFnEZt7M  
Cornell cuda intro : https://cvw.cac.cornell.edu/cuda-intro/       
Tensor cores : https://youtu.be/Yt1A-vaWTck 
Tensor cores2 : https://tgautam03.github.io/2024/10/30/TensorCores/  
tensor cores3 : https://www.glennklockwood.com/garden/tensor-cores    

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
4. blocks : Group / Collection of threads (cuda limits this no. of threads to 1024) make up a thread block 
5. grid : group of blocks in a gpu makes up a grid ( where blocks are arranged ) 
6. Streaming Multiprocessors : the fundamental building block of an NVIDIA GPU, this is where the operations are executed on cuda cores
7. Tiling : A software memory strategy. It is a technique where you break down large datasets into small chunks ("tiles") that fit inside fast, local SRAM (Shared Memory or Registers) to avoid pulling repeatedly from slow VRAM. (memory bound ops with high data reuse)
8. SIMT (Single Instruction, Multiple Threads) means that multiple threads execute the same instruction at the same time, but each thread operates on its own data. Here we need a global thread index , loads 
9. SIMD (Single instruction multiple data) : TODO
10. Tile kernel : level of entire tile block, load a whole tile, perform ops on that tile and store back the tile
11. Single thread : There is nothing as single thread in a GPU execution. even if you define `<<<1,1>>>` its basically initiating a full 32 thread hardware warp to execute a single thread. Just that active mask changes in this case
12. Cuda stream : single stream vs multiple streams TODO
13. sync vs async GPU operations
14. 

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
6. Row major order : So this means how a matrix is laid out in memory, this means rows are laid out besides each other in matrix
7. bump allocation : so we allocate memory in single shot and keep one pointer and keeps on expanding it
8. cudaCheck : Check for if the command that you will be running can that command be run successfully or not and if this fails this immediately prints a failure message and terminates the program. 
 
```cpp
A = [[1,2,3], [4,5,6]]

Row major in memory :
Row memory = [1,2,3,4,5,6]
```
7. Column major order : Means cols are laid out besides each other in a matrix
```cpp
A = [[1,2,3], [4,5,6]]

Col major in memory :
Col memory : [1,4,2,5,3,6]
``` 

#### Thread indexing : 
```
             Columns

          0    1 | 2    3
       ---------------------
Row 0 |   0    1 | 2    3
Row 1 |   4    5 | 6    7
       ---------------------
Row 2 |   8    9 |10   11
Row 3 |  12   13 |14   15
       ---------------------
```

### CUDA language
1. `__global__` : means this function / kernel will run on the device
2. `extern` : declare dynamic shared memory inside a kernel function 
3. Atomics make a single memory operation thread-safe in hardware ( like incrementing a variable )
4. `__shared__` : used to declare shared memory 
5. Mutex makes an entire block of code thread-safe by allowing only one thread to execute it at a time.
6. `cooperative groups` : easy way to allow and manage groups of threads that can synchronize and communicate with each other
7. `__device__` : this means this is a utility function that will run on one of the kernels of the GPU
8. `__shfl_down_sync` : This first syncs / waits for all values to get initialised in thread and then can read its values . Major benefit is it can read value from any thread that is in the same warp without doing an HBM or any other transfer. 
9. `reduce` ops : that takes a list of array of items and combines them into a single value.
10. threadIdx : these are local x,y coordinates inside a thread block ( not global) 
11. blockIdx : this is the thread block idx in a grid
12. blockDim : x, y dimension of a thread block this tells how many threads are present in a thread block 
13. gridIdx : this is a grid of thread block
14. global indexing : `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
15. __syncthreads() : so wait for each thread to reach that line of code / to reach to that instruction
16. coalesced : unified / combined memory 
17. Bank Conflicts : TODO
18. Tensor cores : 
19. Cache line : Smallest unit of data that a computer processor reads from or writes to its internal memory.
20. wmma : warp matrix multiply and accumulate specialized api inside cuda to let programmers talk directly to tensor core hardware

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


Shared memory matmul of matrices 
```cpp
__global__ void matmul2x2(int* out, int* inp1, int* inp2, int N ){
  // inp1 : NxN
  // inp2 : NxN
  // out : NxN
  __shared__ int sm_a[2][2];
  __shared__ int sm_b[2][2];

  int value = 0;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  sm_a[tx][ty] = a[2*ty + tx];
  sm_b[tx][ty] = b[2*ty + tx];

  __syncthreads();
  for(int i = 0;i<2;i++){
    value += sm_a[ty][i] + sm_b[i][tx];
  }
  c[ty * 2 + tx] = value;
}
```


### Tiling vs Warp 
Warp, its a hardware defined execution concept, executes under SIMT ( single instruction , multiple threads) model. So if in a warp a single thread (thread 0) is doing / completing an instruction then thread 1-31 are physically forced to execute that exact same instruction at the instant.


So as the shared memory in a thread block is limited so therefore we need tiling. so block level restrictions are the real motivation for tiling in CUDA.

In tiling, each thread is doing 

```cpp
#define TILE_SIZE 16
#define MATRIX_WIDTH  4096
#define MATRIX_HEIGHT 4096


__global__ void tiled_matmul(float* a, float* b, int N){
    __shared__ float a_tile[TILE_WIDTH][TILE_HEIGHT];
    __shared__ float b_tile[TILE_WIDTH][TILE_HEIGHT];

    int row = threadIdx.x + blockIdx.x * blockDim.x; // output matrix row
    int col = threadIdx.y + blockIdx.y * blockDim.y; // output matrix col
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float dot_prod = 0.0f;

    for(int tile_offset=0;tile_offset<N;tile_offset += TILE_WIDTH){
        // check boundaries
        int a_check = (tile_offset + tx < N) && (row < N); // col that we are using in this 
        a_tile[ty][tx] = a_check ? a[row * N + tile_offset + tx] : 0.0f;

        int b_check = (tile_offset + ty < N) && (col < N);
        b_tile[ty][tx] = b_check ? b[col + N*(tile_offset + ty)] : 0.f; // to check  
        // here we are assuming row major indexing that is : 
        // row = tile_offset + ty; (cause the indexing here is done in this x,y manner that is same as the axis that we follow in maths)
        // col = col
        __syncthreads();

        for(int i =0;i<TILE_WIDTH;i++){
            dot_prod += a_tile[ty][i] * b_tile[i][tx];
        }
    }

    if(row < N && col < N){
        c[row * N + col] = dot_prod;
    }
}
```


## Tensor cores 
Introduced in 2017, Specialized physical ALU's designed for matrix multiply and addition (MMA) operations. 
And these are new so they only support few matrix sizes like `16 x 16` and others 
these are also just tiled matrix multiplication and nothing else


## Coalesced memory access
Combining global memory accesses from some/ all threads in a warp to a single memory operation. Easily achieved when overall request brings in data from consecutive memory addresses starting on a good memory boundary. 

So we have row major and col major memory access patterns and based on how you are accesing it / defining threads you can improve it 

```cpp
#define WIDTH 4096 
#define HEIGHT 4096

// col major
__global__ void col_order(int* a, int* b , int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int col = i % WIDTH;
  int row = i / WIDTH;
  int idx = col * HEIGHT + row; // cause here we are using this method to acces memory and this access it col major access

  if (idx < N){
    a[idx] += idx;
    b[idx] += idx;
  }
}

// row major
__global__ void row_order(int* a, int* b , int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col = idx % WIDTH;
  int row = idx / WIDTH;
  
  if (idx < N){
    a[idx] += idx;
    b[idx] += idx;
  }
}
```

## Bank conflicts 
Shared memory is divided into equal sized memory modules and if each thread in a warp accesses a different bank then all memory transfers can be done in parallel. 

CUDA configures bank using this : 
NO_OF_BANKS = 32  
`bank_index  = (address/word_size) % NO_OF_BANKS`

No. of banks is fixed at 32 matching the warp size. word size is 4 bytes by default
So if we are using something like this : `shared[threadIdx.x * 2]` then addresses become 0,2,4,6 .. and now 2 different threads access Bank 0.
A bank has only one read port and it cannot fetch both words ("word" means one addressable memory unit)


```cpp
#define OFFSET 33 // different offset can cause more / less conflicts
__global__ void func(int *a, int n)
{
    // shared memory
    __shared__ int val[1024];
    int tid = threadIdx.x;wa
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int c = tid * OFFSET % 1024;
        val[c] = a[idx];
        a[idx] = val[c] + 1;
    }
}
```

## Tensor cores 













