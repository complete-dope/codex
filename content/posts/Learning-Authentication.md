---
layout : post
date : 2025-03-06
title : Learning Authentication 
---

# Learning Authentication and Tokens 

So when using an auth library like Auth0 we get 2 tokens : 
1. JWT ( JSON web token )
2. Access token

What's the use of these 2 and not just use 1 ? 

ID Token is all about who the user is , the details , identity of the user , user's unique identifier

Access Token is your key to authorization. It tells the resource server that the bearer of the token has permission to access specific resources. 

API authorization : when your client application makes requests to a protected API, it includes access token ( typically in HTTP authorization header ). 


So we need to verify identity of the user at the backend also, that being said, the frontend already validated it, so we take the token from the frontend and then validate the user from there ! 


Frontend sends the access token to the backend in form of 'Authentication : Bearer <access-token>' 

Verification process : backend uses a special library to decode and verify the token, that includes signature check , claim check and then access granted or denied.


If u need the details of a user then we need to use ID token 
Then for subsquent request I will verify it using <access-token>

Just by using the access token we can also get the user details ,the endpoint for the same is, `GET /api/v2/users/{id}` using this
[Get user data from access token](https://tus.auth0.com/docs/manage-users/user-search/retrieve-users-with-get-users-by-id-endpoint?utm_source=chatgpt.com)

Once we get the access token, we need to decode that jwt and get the user_id that is in the payload and from there we can verify the user !!    







