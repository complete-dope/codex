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

  

