---
layout : post
date : 2025-03-07
title : Learning internals of building a PC
---

[computer from starting up](https://cpu.land/the-basics) <br>
[computerphile explaining CPU](https://www.youtube.com/watch?v=IAkj32VPcUE) <br>
[Inside a CPU](https://www.youtube.com/watch?v=iNfuE3MLmzA) <br>
[blog explaining cpu working](https://www.trentonsystems.com/en-us/resource-hub/blog/what-is-a-cpu) <br>
[compiler explorer](https://godbolt.org/) <br>

# What is a disassembler? and what does that mean? 
Disassembles Machine code to human readable assembly code and the assembly code is specific to the processor as it depends on ISA of the processor !! 
Popular disassembler is IDA ( integrated disassembler ) 
Modern compilers add a lot of functions to make it efficient so its becomes an art decompiling it !!  


# How does a CPU works ?

So this all starts as I wanted to learn how a GPU works and without knowing CPU indepth I wont be able to appreciate GPU enough !! So here we learn how computers work we will write some assembly code , see the flow how that turns out in CPU instruction set and the optimisations made in CPU to reduce time taken !

CPU the most important part / brain of a computer ! 
It has 3 parts that are : 

1. Fetch the instruction from memory
2. Decode the instruction
3. Execute the instruction 


## Fetching the instruction 
Lets say we want to print character on a screen, that in assembly would look like :

```asm 
.LC0:
        .string "Character is "
printme(char):
        push    rbp
        mov     rbp, rsp
        sub     rsp, 16
        mov     eax, edi
        mov     BYTE PTR [rbp-4], al
        mov     esi, OFFSET FLAT:.LC0
        mov     edi, OFFSET FLAT:std::cout
        call    std::basic_ostream<char, std::char_traits<char>>& std::operator<<<std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>>&, char const*)
        mov     rdx, rax
        movsx   eax, BYTE PTR [rbp-4]
        mov     esi, eax
        mov     rdi, rdx
        call    std::basic_ostream<char, std::char_traits<char>>& std::operator<<<std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>>&, char)
        mov     esi, OFFSET FLAT:std::basic_ostream<char, std::char_traits<char>>& std::endl<char, std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>>&)
        mov     rdi, rax
        call    std::ostream::operator<<(std::ostream& (*)(std::ostream&))
        nop
        leave
        ret
```

Here the one's starting with 'r' are the registers !

The fetching stage means, using the address bus ( a 16bit bus / connection that sends the memory address to the CPU ) we fetch the instruction at that memory location .. The memory location is usually a 8byte memory stack laid out !! (If its RAM, then its kinda incorrect !!)

Address bus sends the address to the memory 
Data bus sends back the data to the CPU (which initiated it in the first place) 

## Decoding the instruction ( mention examples in this )
We need to decode the instruction, once we get that instruction from the fetching part we need to then decode it to actually make it usable to the CPU,


## Executing the instruction ( mention examples )
Once decoded we need to execute / carry out that instruction ( i.e. apply maths , wait, get data .. etc) and storing 

## Clock speed ( what ??)


## Threading ( how does that work with examples in assembly) 

In `python` :
OS Threads are spawned to achieve concurrency but due to GIL, at a time a thread from a single python process can only run.. so lets say you have 4 cores , and your python process uses , 100 threads so at any given time the process you started that main thread + 100 subsequent one that your process spawns out of these only 1 can be executed even if they are all independent of each other .. that how GIL works

You can do multiprocessing but that doesn't solve the threading problem you have above 

![GIL](https://github.com/user-attachments/assets/1253e5c2-0e61-46e1-8331-c7ff9539f64b)

Here your process even though started multiple threads GIL is eventually locking them down to use one thread at a time ... 


In `golang` :
You create Goroutines , worry nothing about threads, the go will map using M:N mapping , OS threads to goroutines and uses all cores, actual threading to get the work done .. true threading / concurrency

## Event loop 
In python, Single threaded, process / event loop , that stacks up the python asynchronous calls , IO calls , sleeps etc. Its unaware of threads that are running , can be added in the same event loop using `loop.run_in_executor()` 

So what that means is, your python process and its main event loop is running fine, if I start a new thread then its an os level thread, event loop will not wait for it to end, that is, that thread will run in bg till the python process is running, and request will be returned as soon as event loop ends so its a win-win.


## Concurrency 
Its the ability to do multiple tasks one after another in such fast manner that it seems that many tasks are getting executed at once and but in reality , at any given time , a single core is working with a single task ..
Task1 picked up , waiting for I/O , add back to queue start with task 2, waiting for user IO , add back to queue , start with task 3 .. this is how CPU works and that too in clock cycle .. this is called context switching and to remeber from where it has to continue it uses : Program Counter , Stack pointer, General purpose registers  

## Parallelism 
Its a subpart of concurrency , where in reality multiple tasks are getting executed, that can be done when each core is doing a different task. Parallelism is one way of acheiving concurrency. 

## Multiprocessing 
Its also a type of concurrency 
Now the machines have multiple cores, so that means, 

In `python` :  
Run each code snippet on a different Core so that we can tell CPU thinks its running so many process (it is but they are the same processes) and each code snippet has its memory space and variables defined ...   
Using multiprocessing library each core now runs the same code on each core that leads to high memory use and higher file sizes, that makes context switching expensive (as memory space also needs to be loaded)

This is useful to run same python process from each core ,but if each process spawns up its own threads then its not useful .. as at a time one thread per python process .. 


In `golang` :
[goroutines](https://jayconrod.com/posts/128/goroutines-the-concurrency-model-we-wanted-all-along)



## Cores
The cores in CPU are just physical partition at the silicon level ( if it was theoretical people would have exploited it more ) .. so that a single process (aka a long task) dont take up the whole computation / processing / CPU and we no more operation can run on it ... 
So cores help in managing overall working of a CPU ( that might involve, sending data to GPU , showing viz , storing , fetching etc ) 

Each core has threads for it and there role is to pick up a task send that to a seperate core or same core (often on a seperate core ) and bring that data back from the core once the computation is all done and return back !! 
Kinda like a worker and we call them workers  

Processes can initiate / create more multiple threads from there process and to they can all run independently on different core and get the work done , but python limits them using GIL (as explained above)


## Modern day server

Event driven : 
Single main threads runs over , requests comes in they are all queued in the sockets uses `select` syscall.  select accepts a list of file descriptors (generally sockets) and returns which ones are ready to read or write. So these are event driven architecture and works in nodejs as well ... 

[servers](https://jayconrod.com/posts/128/goroutines-the-concurrency-model-we-wanted-all-along)


## Die shots 

`NOT COMPLETING IT !! LEXI HAS WRITTEN WAY TOO GOOD IN CPU LAND CANT BECOME BETTER !`

[SIMD](https://www.youtube.com/watch?v=ulmjqD6Y4do) <br>
[Improving performance](https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/) <br>
[x86 arch](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/x86-architecture) <br>
[x64 arch](https://learn.microsoft.com/en-us/archive/msdn-magazine/2006/may/x64-starting-out-in-64-bit-windows-systems-with-visual-c) <br>

## How to program a CPU ?

Single Instruction Single Data (SISD) <br>
Single Instruction Multiple Data (SIMD) <br> 

`Question : Writing instruction for a CPU in C lang (coding) vs writing SIMD in C how does that matter ? arent both the same?`

In C, you depend on the compiler's to convert to machine code ( u trust too much on compiler ) that will do it optimally and place in apt registers for faster  

In SIMD in C, uses SIMD registers that basically fakes CPU to do multiple operation on multiple data at a single time, this is called vectorisation, given the processor supports AVX (Advanced vector Extension)!

Example : 

C code 
```c
for (int i = 0; i < 1000; i++) {
    array[i] = array[i] + 1;  // Adds 1 to each element in the array
}

```

SIMD code :
```c
for (int i = 0; i < 1000; i += 8) {
    __m256i data = _mm256_load_si256(&array[i]);  // Load 8 elements at once
    __m256i result = _mm256_add_epi32(data, _mm256_set1_epi32(1)); // Add 1 to each of 8 elements
    _mm256_store_si256(&array[i], result);  // Store the result back to the array
}

```


`Question : Why does `for` loop inherently use this SIMD registers if they are so good ? Why do we have to tell them to use SIMD now ? Cant it identify its a SIMD operation ? `

TL;DR : Compilers are dumb in 2025
Modern compilers can auto-vectorize loops and convert certain code into SIMD instructions if the target architecture supports it (like AVX). However, it's not as simple as "just do it" for a few reasons:
1. Compilers are cautious. They only auto-vectorize loops when they're absolutely sure it won't change the program's behavior.
2. SIMD instructions often require data to be aligned in memory.
... 

Each processor / Central processing Unit has a ISA ( Instruction set archicture ) that defines the registers , formats etc (In general, an ISA defines the supported instructions, data types, registers, the hardware support for managing main memory,[clarification needed] fundamental features (such as the memory consistency, addressing modes, virtual memory), and the input/output model of implementations of the ISA. )  which tells all program that are gonna run in the CPU after getting compiled to Machine Code, that look I am expecting instructions in this format you dare not give me something else !!

Intel has CISC isa , that is defined seperately !!
Mac and phones uses RISC , that tells that I can receive instructions in this manner !!

[Processor registers](https://en.wikipedia.org/wiki/Processor_register) <br>
[Multi Media registers](https://stackoverflow.com/a/44299695) <br>


SIMD (concept) and inside it we have an subset called as XMM registers ( they constitute SIMD ) !!
AVX (Advanced Vector Extensions) : A SIMD instruction set within the x86 CISC architecture.
The term x86 comes from Intel's 8086 processor, which was the first in a family of processors with a similar instruction set.

x86 processor -> uses CISC isa -> introduced SIMD concept 
SIMD in x86 (earlier aka SSE) -> registers XMM
SIMD in x86 (now aka AVX) -> registers YMM , ZMM


MM registers 
This was a latest addition to intel x86 isa to bring the concept of SIMD , and have now been upgraded to XMM , YMM , ZMM ( each increasing the no of bits we can store for parallel processing data)   
Using 





 ### `Windows games cant run in linux without conversion ? why ? even if i use same processor ?`
This is because the games developed by a windows developer uses windows DLL and many more , and to use them on Linux machine would require to change them in linux file format (ELF) and many more
So we require a compatiblity layer for this conversion at the software level !!

where ever software layer comes things become a bit sorted / easy for developers to test it out !! 

# How does a GPU works ?  

Writing another article for this !! 








