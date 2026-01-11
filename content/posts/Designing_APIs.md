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

# Transformation 

Some requirements that we already have should be transformed as needed, so things like query parameter like `/bookmarks?page=2&limit=10` 
but all query parameters are in the string type by default so we need to cast it before passing to validation pipeline

 
