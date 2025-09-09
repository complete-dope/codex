---
draft: false
title: Async Programming
date: 2025-01-02
---


[Google doc link](https://docs.google.com/document/d/1gMVzz5Haivst864_JZKWwY_AiNvu6pxMpttg2-dJopk/edit?usp=sharing)
[Async Programming](https://www.blog.pythonlibrary.org/2016/07/26/python-3-an-intro-to-asyncio/)

# Basics of Async Programming 

At any single time, a single CPU thread executes only a single function ( no multiprocessing happens ) atleast in the case of python !! Its just smart switching 

1. Pick up a event loop / create a event loop
2. Create / Add tasks in it 
3. Check for the status and if done return the result 

## Coroutines : 
A coroutine is a special function that can give up control to its caller without losing its state. 

* Who's the caller here? 
* Event loop 

Functions that can suspend the execution and retain there state variable and resume back 

`yield` statement 

`yield <output>` outputs value to a function willing to accept it using yield statement
`output = yield` function that waits for input to come and till then gets suspended 

It stores the local variable in the `<coroutine_obj>.gi_frame.f_locals` here we can see the states in a dict format !


## Future

Task is a subclass of future ! 
A Future is an object that represents the result of an asynchronous operation that hasn't completed yet. Its a promise that something will happen in future 

Think of a Future as a pizza order receipt
* Pending: You've ordered but it's not ready.
* Done: Pizza is ready, and you can pick it up.
* Cancelled: You decided to cancel the order.

## Task 
When the future has started to run then its called a task.
Directly schedule a task using this async method 

## The use of async await coloring 

Async makes the function async that is execution could be suspended and when we do await then (if it asynced till the very end aka network call) the event loop leaves this task and takes up the other task that is has in its event queue .. it cant pick up the call that is present in the next line because it has left its execution of that particular running item its takes up its mostly backlog requests which are queued in that event loop.. 
See this : 

```python

async def wait_2_sec():
  await asyncio.sleep(2)
  return "slept 2"

async def wait_3_sec():
  await asyncio.sleep(3)
  return "slept 3"

async def task_1_sec_sleep():
   await asyncio.sleep(1.5)
  return "completed task that took 1.5 sec" 

async main():
  coro1 = task_1_sec_sleep()
  task1 = asyncio.create_task(coro1) # backlog task 
  st = time.time()
  await wait_2_sec()
  # while its waiting it will complete this backlog task1 here ( so that is the use of async await )  
  await wait_3_sec() 
  et = time.time()
  print("total time ", et-st)

# run this in async loop
main()
```

### How chatGPT answers Streaming response

```python 
import time, string

def llm_output():
  time.sleep(1)
  val =''.join(random.choices(string.ascii_lowercase, k=15))

  # val = random.random()
  return f"Tokens-{val}"

def generate():
  while True:
    output = llm_output() #passing to UI
    yield output #produces output     

gen = generate()

def response():
  isEnd=0
  while isEnd != 1:
    value = yield #tuple
    print(value)
  pass

res = response()
next(gen)
next(res)

while True:
  output_tok = next(gen)
  res.send(output_tok) 
```

### Generator :
Generates output, so a function that periodically generates output 

```python
def _generator():
  print("Generator, ")
  while True:
    yield "Generates Final"

gen = _generator()
next(_gen)
```

### Receiver
Receiver input, that receives the input

```python
def _receiver():
  print("Receiver, it takes the data")
  value = yield

recv = _receiver()
recv.send("Value-1")
```


## Using the async-await keywords !! 
For async to work we need to make sure that we make it async to the lowest level / OS level otherwise nothing is asynchronous .. only on surface level its asynchronous !!  


```python

import asyncio
import aiohttp


async def counter(idx=1):
    print(f"Counter {idx} started")
    await asyncio.sleep(idx)
    print("Exiting the counter")

async def fetch(idx, url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"{idx}-{url} → {response.status}")  # Print status when done

async def main():
    await counter(2) # must finish before going ahead
    await fetch(10000, "https://jsonplaceholder.typicode.com/users") # wait for this one before all others
    urls = ["https://jsonplaceholder.typicode.com/users"] * 10  # 10 URLs ## SCHEDULED CALLS 
    tasks = [fetch(idx,url) for idx,url in enumerate(urls)]  # Create tasks
    await asyncio.gather(*tasks)  # Run all at the same time

asyncio.run(main()) # creates a loop, executes , closes
```


Await just means you need to wait for this operation / function to complete before moving ahead !!  

This makes sure that the counter and 10000 call's are finished before you start anything new !! 

You can make functions async by clubbing functions that you want to run together ... if you will just keep a await statement on every function that you call then it again makes it async call only !! OHH !!

Means the one we are using in project its just asynchronous !! 

So I need to test whether I can gather and run and then again start with below part ? 

```python
import asyncio

async def fn1():
  pass


async def fn2():
  pass


async def mid_main():
  asyncio.gather(fn1() , fn2())


asyncio.run(mid_main()) ## HERE WE RUN THE FUNCTION 


async def ahead():
  pass


async def ahead2():
  pass

async def fin_main():
  asyncio.gather(ahead(), ahead2())


asyncio.run(fin_main()) ## HERE WE RUN THE FUNCTION

```

Yes we can do this !!

```python
import asyncio
import aiohttp
import time


async def counter(idx=1):
    print(f"Counter {idx} started")
    await asyncio.sleep(idx)
    print("Exiting the counter")

async def fetch(idx, url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"{idx}-{url} → {response.status}")  # Print status when done

async def main():
    await counter(2) # must finn
    start = time.time()
    await fetch(10000, "https://jsonplaceholder.typicode.com/users") # wait for this one before all others
    end = time.time()
    print("Time for a single API call is " , end - start)
    urls = ["https://jsonplaceholder.typicode.com/users"] * 6  # 10 URLs ## SCHEDULED CALLS 
    tasks = [fetch(idx,url) for idx,url in enumerate(urls)]  # Create tasks
    await asyncio.gather(*tasks)  # Run all at the same time
    print("All API call time is ", time.time() - end)


asyncio.run(main()) # creates a loop, executes , closes

async def main2():
    await counter(3) # must finn
    start = time.time()
    await fetch(100 , "https://fakestoreapiserver.reactbd.com/smart") # wait for this one before all others
    end = time.time()
    print("Time for a single API call is " , end - start)
    urls = ["https://fakestoreapiserver.reactbd.com/smart"] * 10  # 10 URLs ## SCHEDULED CALLS
    tasks = [fetch(idx,url) for idx,url in enumerate(urls)]  # Create tasks
    await asyncio.gather(*tasks)  # Run all at the same time
    print("All API call time is ", time.time() - end)

asyncio.run(main2())

```

Async example-2

```python
import nest_asyncio
nest_asyncio.apply()  # Required to avoid event loop conflicts in Colab

import asyncio 

class Main2():
  async def async_fn_from_main2():
    await asyncio.sleep(5)
     

class Main():
  def __init__(self): 
    a = 10;
    b = 20; 

  async def asyncfn1(self):
    await asyncio.sleep(4)

  async def hello(self):
    await asyncio.sleep(5)

  async def main(self):
    task1 = self.asyncfn1()
    task2 = self.hello()
    task3 = Main2.async_fn_from_main2()
    await asyncio.gather(task1 , task2, task3)
    print("got the data") 

asyncio.run(Main().main())

```

EXAMPLE MAKING HTTP CALLS ASYNC - 3 

```python

import nest_asyncio
nest_asyncio.apply()  # Required to avoid event loop conflicts in Colab
import aiohttp 
import asyncio 

async def api_call_1(session):
  # async with aiohttp.ClientSession() as session:
  async with session.get(url) as response:
    response.raise_for_status()
    return await response.json()

async def api_call_2(session):
  # async with aiohttp.ClientSession() as session:
  async with session.get(url) as response:
    response.raise_for_status()
    return await response.json()


async def main():
  async with aiohttp.ClientSession() as session:
    a = []
    for _ in range(100):
      a.append(api_call_1(session))
      a.append(api_call_2(session))

    dd = await asyncio.gather(*a) 
    return dd

st=time.time()  
d = asyncio.run(main())
et = time.time()

print("total time" , et-st)
print(d)

```


**METRICS**

Typical Linux System	1000–65000 (depends on tuning)

Python asyncio + aiohttp	1000–10000 (efficient)

Thread-based (e.g., requests)	~100–500 (less efficient)

Nginx Default (worker_connections)	1024–4096 per worker

So , approx 100-200 connection can be made parallel from the same OS !! 



## How does this work in the low level ? 

The OS works in the following manner, network request comes from the python application, then it makes requests to the OS , then OS calls networking library, socket it created in a non-blocking manner and upto ~60000 TCP connections can be made .. but in reality we can have around 100 network calls at max. 

and it the epoll tells the OS when is the data received and then it tells it back to the application .. 






## Django server stats

**1. Code: async, Server: sync (WSGI)** → Result: async
Under WSGI, Django creates a one-off event loop in your thread.

Your async def view with awaitable calls (like HTTP requests) can interleave tasks inside that loop.

✅ You get intra-thread concurrency, so things run faster—your logs are valid.

**2. Code: sync, Server: async (ASGI)** → Result: sync
ASGI offloads your sync view into a thread pool using sync_to_async.

It still blocks that thread; no benefit from event loop.

So performance stays sync, not async.

**3. Code: sync, Server: sync (WSGI)** → Result: sync
Classic model—thread-per-request, fully blocking.

No async anywhere.

**4. Code: async, Server: async (ASGI)** → Result: async
True async all the way: event loop, async view, awaitable I/O.

Best throughput, non-blocking across requests.



Built in server in django is multithreaded , sync server .. that means each request spins up a 1 seperate OS thread and gets executed by the individual request loop
Async server ( uvicorn ) makes it multithreaded , async server .. that means all the requests are scheduled in the same Event loop as async task by the process level event loop (Concurrency occurs via await yielding , waiting for async tasks to complete)


### Terms from streams 

Async requests have to be async till the very end .. (that's called leaky abstraction)

`function coloring ` : This means that we are making some functions colored as sync and other as async, and this coloring based differentiation is being made out 















