---
draft: false
title : Tech terminologies that makes you a pro-pro developer
date: 2025-04-21
---

### Serialization : 
To convert / store data in a compatible format that can be used later. Means lets say I want to store a list of dictionary `[{'key1' : 'val1'},{'key2' : 'val2'}]` in redis .. so there is not data structure supported in redis for this so only way to store this is to serialize it into bytes to store and then to retrive back we need to deserialize this .. 
How to do this : 
`json.dumps()` : convert the dictionary to json string then store it in bytes
`json.loads()` : converts back the serialized object to the original state by deserialization     

| Feature              | `json.loads()` | `ast.literal_eval()`         |
|----------------------|----------------|-------------------------------|
| Format               | JSON           | Python literal expressions    |
| Accepts single quotes| ❌             | ✅                            |
| Tuples/Sets          | ❌             | ✅                            |
| Safe to use          | ✅             | ✅                            |
| Executes code        | ❌             | ❌                            |
| Handles full Python  | ❌             | ❌ (only literals)            |


### Literal :
A literal is just a fixed value written directly in your code — not something computed, called, or looked up.









