---
layout : post
date : 2025-03-06
title : Learning Authentication 
tags : ['authentication', 'authorization', 'session', 'jwt', 'cookies']
---

# Learning Authentication and Authorization

HTTP in itself is stateless that is each request is considered as a seperate new request and with no previous baggage or memory.  

3 basic terms : 
1. Session 
2. JWT
3. Cookies

JWT : JSON web token, 64 bit cryptographic hashed token, has 3 parts seperated by '.', those are `headers`, `payload`, and `server side secret-key` , so if anyone tries to change any value in any of the fields the hash will change and will no longer remain valid and our server side secret wont be able to match it (uses minimum resources to validate, only a single secret string that can be stored in env also ) , this JWT can be stored in a cookie, local storage , memory etc

Cookies : These are also a storage way that servers use to store a session-id or something that can later be used by server and the storage and management is done by browser (these are one way of doing it ,other that we already discussed is the JWT), so with every request to that domain they are shared   

## Authentication
Who you are ?

So session based authentication is a stateful method, that is you enter your details, these details are stored in a db like redis, and in return you get a session-id / user-id in a form of JWT or in a cookies that server takes to recheck / revalidate the user. 

4 primary methods

* Stateful : where auth management is done by server ( like session based )  
* Stateless : like JWT, where a token is self sufficient to authenticate   
* API keys : that is used for server to server communication (that is one server requiring access to another server core functionality)  
* OAuth 2.0 : a method for applications to connect to each other   


JWT are send if each request in an Auth Header and these are used as access token (if payload contains a field like role / scope) , Bearer token ( Added as an auth header)  

OAuth, so lets say you want to connect to google calendar to schedule some meeting how to do that ? 
Delegation : Apps requiring to connect to each other is called Delegation   
> earlier people used to share password but that is highly inefficient what if you shared that to Untrusted site  

Better method is to have an Auth server that creates a token that signs in with role ({'scope' : 'read, update, delete calendar events' , 'expires': '1hr'}) , and then you show this token to the request server and now you can make changes there.   
In the above case the `auth-server` is google and the `request-server` is also google.   


# Authorization 
What all can you perform in this ? 
You can do these things, defines a scope / value that a user can do with this application, so how to identify a role user / admin / developer / clients from this , can be signed into JWT also

## Some things to keep in mind 
Dont be verbose with the output results from the server, always says generic like 'Authentication failed'   
Timing bomb : Give output after a delayed time only so hacker cant tell where the request got failed   







