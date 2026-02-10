---
title : Designing APIs from first principles
description : Designing APIs 
date : 2026-01-12
tags : ['apis']
---

# HTTP designs 

Options list what all options are present for this host method , (POST, GET , PUT, PATCH, DELETE) 

# API designs 

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




