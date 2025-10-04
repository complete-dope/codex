---
draft: false
title: Learning Event Loops ( JS / Python )
date: 2025-01-26
---

# Learning JS Event Loop

[Event loop](https://www.youtube.com/watch?v=cCOL7MC4Pl0)


Main thread : where every thing happens 
something on main thread, takes a long time ( 200ms ) that becomes noticeable to the users and is not user exp

The human body , is multithreaded in all ways , we can type , listen , understand , talk , move eyes , move lips ..etc all at the same time 


But only while sneezing, we become single threaded, we cant do nothing but sneeze 

so what we do is create multiple threads (networking thread , crypto thread , monitoring IO device etc ) and once they are done they have to combine back to the original-main thread

Event Loop is the one that orchestrates all of this !!

What would happen is lot of JS running parallel in many threads and all editing the same DOM , and you end up with all these race conditions 

so we queue the taks and run that in a step 

## Event loop
Runs around CPU in an efficient manner, tasks get schedules it picks up efficiently completes it 
Each application / Main process has its own event loop  

### Task Queue
When we queue a task, the event loop takes a detour, browser says to event loop I have got a job for you to run .. event loop says add it to my to-do list 

Only one task can be processed at a time, rest all the tasks get stacked up and wait for there chance to play  

To queue a task we can use `setTimeout` in JS  

The rendering calculation happens at the start of each frame , that includes style calculation , layout and paint  

### MicroTasks
associated with promise 

Browsers wanted to give developers a way to monitor DOM changes 
Using DOM we change the HTML in a programmatic way and all langs do that .. 


JS task:
If from some button click , an event got registered and that event initiated a microtask, then the JS will first finish task on stack, then call microtask then call next JS function ...   


## EVENT LOOP IN  Django framework

We can run the django event loop on local-server , the built-in server in which the model runs using `python manage.py runserver` that is a WSGI server aka synchronous server.
That means event loop starts as soon an async view comes and end when the request is completed and the request is logged as completed once we `return HTTPResponse()`.

So to any event that is added to return loop after that get's autocancelled / destroyed. 

To overcome this the classic thing to do is : to start a new loop, seperate from the current execution loop, and add your event there and close the loop seperately 

Cases: 
1. New loop -> new thread  : doesnt interfare with existing functionality 
2. Same loop -> new thread : interfares  with its own loop
3. Parent thread can cancel execution , no python doesnt provide a built-in / safe way to identify the child threads and cancel it  

When we run our django app in production behind an ASGI ( async server ) the event loop is not destroyed for each request and its managed by the async server. 


Uvicorn works at the process level not per request or per thread. 
Each worker process runs in a single event loop and the loop stays alive for the entire process lifetime and this is how celery also works  


Runserver is multithreaded by default , that means if spawned 10 threads and each thread requests the server if works concurrently ... 
But why cant this be mimicked by async requests? 














