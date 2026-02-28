---
title : Reliable / scalable Backend 
description : includes queues, balancers , low-level design , high-level design, designing APIs 
date : 2026-01-12
tags : ['apis', 'lld' , 'hld' , 'balancers' , 'queue']
---


## Terms 
1. Ephemeral (efimeral) : short lived functions like google-cloud functions , that are running for per request and has a fixed timeout and at that timeout it sends a SIGTERM to close it.

2. Presigned / Signed URL : Is used for upload / download directly to storage (w/o involving backend in this all), its a time based URL that expires with time, the generator of the link should have access and uploader can just upload ... so this is used in Customer support as well , when we upload images of a damaged item, this is used to reduce load to backend and solve this   

3. Serialization : 
To convert / store data in a compatible format that can be used later. Means lets say I want to store a list of dictionary `[{'key1' : 'val1'},{'key2' : 'val2'}]` in redis .. so there is not data structure supported in redis for this so only way to store this is to serialize it into bytes to store and then to retrive back we need to deserialize this .. 
How to do this : 
`json.dumps()` : convert the dictionary to json string then store it in bytes
`json.loads()` : converts back the serialized object to the original state by deserialization     

4. Hydration : The process of "filling in" a data structure (like a feature vector, a request object, or a model input) with its full set of required data. So when a data comes in from an API we usually pass that data through some data structure for schema verification that part is called hydration  

## Event driven architecture
This works in a queue based system, that is, we have a producer that is doing some operations and sending that to broker (kafka / mq) and then on other side of message queue are consuming it. 

These queue are usually designed for kb-mb of message sizes only 

and assume we get 1000's producers producing and sending messages to brokers, kafka handles this internally using `epolls` that sleep when nothing is producing and as soon as we get somethign on network stack it does an NIC interrupt and wakes up a sleeping thread and that is passed on to the CPU for processing 


# API designs 
Options list what all options are present for this host method , (POST, GET , PUT, PATCH, DELETE) 


## Validation 

Firstly we should have a validation layer that should be very clear and any request that is not validated should raise this error so that its clear to client  what's expected

1. Syntactic : Validations like if the email is even an email or not , phone no. is 10 digits , name is a string with length > 2 character , dates in a structure etc

2. Semantic : Data makes sense or not , that is date-of-birth should not be 2050 ( in future)     

3. Type Validation : name should be a string, timestamp should be a number

This should be followed by a 400 request code that should be a bad request 

## Transformation 

Some requirements that we already have should be transformed as needed, so things like query parameter like `/bookmarks?page=2&limit=10` 
but all query parameters are in the string type by default so we need to cast it before passing to validation pipeline


# Middleware , Binding and Lifecycle 
Lifecycle the entire end to end cycle that a web request goes through its called a lifecycle from the moment a request comes in to the final response time that is called as HTTP response lifecycle

Middleware : this are function that sit between request to the final response preparation, 
A normal handler includes (request, response) , that is request input and response is where we write this and send back to the user. 
A middleware takes in 3 things that are (request, response and next) these are just functions and we can multiple middlewares. 

REST API designs 

> follow the standard so that people all around the world can use it

* GET for fetching some document from server  
* POST for making a document / doing everything besides CRUD operations
* PUT for creating / adding something to DB 
* PATCH for fixing some data that was already added before in db or somewhere
* DELETE for deleting a resource


Before a browser calls any API, it sends a preset fetch request that tells what all can be extracted or is there is an CORS issue in this request. 

## Webhook 
There are few problems in doing a webhook call request,
you cant check that in localhost if your server is deployed (you need to call call something that is available from public internet)

And to segregate this a possible way is to change server code and check a condition when to call (very doable, possible , best approach )   
Another way , if the webhook calls are dynamic then pass this as a request param   



## Authentication 
Refresh token : this is a token used to get a new access token  
Access token : short lived JWT (15 min) and they are passed to auth server when frontend starts getting 401, and auth server returns a new refresh token and a new access token that is used for next 15 minutes, this is how its in web most time
HMAC : Hash-based Message Authentication Code, this is like JWT, so generate a hash on both servers and match it ( obvio add this in header as well ) 

## Timeout 
Always add timeout to API, so if a request fails , raise a ConnectionTimedOut error and change status to False 


## Making multiple requests and next one depends on prev one 
These all cases are handled via sending that in the request headers to make sure someone didnt injected that in between and to make sure this is an consistent request .. add in headers for handling auth for request. These are called signed URL , that means this URL is signed by someone that has the SECRET-KEY .   



# Design patterns 

1. `Strategy Pattern` : Think of this as a game strategy , the underlying input that is the game that is same for all strategies it's just that the implementation is changed so liek that the design pattern that we can use for all sorting functions

2. `Iterator Pattern `: dont tell the underlying Data structure like tree , linked-list , etc just use an `__iter__` and `__next__` method to sort this out.

3. `Adapter Pattern` : So this is bridge between 2 code base , the client wants in a specific format and you have implemented in other format so its a kind of adaptation from old code to a new code




