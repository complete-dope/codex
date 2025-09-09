---
draft: false
title: Learning Basics of networking
date: 2025-01-02
---

## Basics of Networking

[Google doc link ](https://docs.google.com/document/d/1dQU7fJDfAWogpLl-JptT1eaHzGoXHpcx_y3KVeejCAc/edit?usp=sharing)


### SSH explained 

`ssh` is a secure way to connect to a machine 

**Logic** 
A 2 layer key is setup , a public and private key you 

**Setup** 
Lets say a machine in London wants to talk to a machine in Delhi , london machine is connected to wifi and has a fixed internal IP ( 192.168.1.200 ) , the wifi IP is Dynamic , so in the router's setting we need to create a port forwarding rule to internal IP (192.168.1.200) on port 22

and on the Delhi machine we need to turn connect using: 
`ssh -p 2222 your_username@Public_IP` : This tells ssh to connect to router's public IP     

**Working** 



### TCP explained ( Transmission control protocol )

A protocol to send data over internet its nothing just a reliable method and ensures reliability of data. 

**Logic and Example** 

Example: 
Lets say you want to share 10MB of data from your device on internet
The OS just hands over it to networking stack that has TCP built into it 

TCP breaks it into smaller chunks called segments. 
Labeling, each segment gets a unique sequence number
IP layer to provide the headers to the data 

The Data Link layer (handled by your NIC and drivers) and the Physical layer (your network hardware, e.g., Ethernet, Wi-Fi).

**Setup**



**Working**

## Protocols 
HTTP protocol was made by internet engineering task force , and its a document that explain this protocol and rules of how to use it .. 
Set of rules like language , so any 2 parties that follow these rules or speak this language can talk to each other 

HTTP is made on top of TCP, (OSI model) , the syn , syn-ack , ack this happens over TCP protocol 

### RPC vs HTTP 

Learning about rpc , grpc , http and alternatives

rpc : remote procedure call  

client calls a function, the function is located on the server, the server then returns the value back to client

marshalling is same as serialization i.e. conversion of objects into format suitable to transmit like bit-stream 

unmarshalling is same as deserialization i.e. reconstruct to original form from serialized form 


`Stubs` : stubs or proxies that are generated from Interface definition language , RPC system convert these remote execution over the network .. ( not clear yet ) 



![rpc-example](https://slideplayer.com/slide/8976075/27/images/15/Example+of+an+RPC+No+message+passing+at+all+is+visible+to+the+programmer..jpg)


client converts to stub -> sends the stubs over the network using ( http/2 , tcp ) then the server recreates this stub , calls the actual function and return backs the data


* Internal system : this means microservices talking to each other, each having its own container and these services like inside the same kubernetes cluster ..


rpc is better than rest for internal system  call : 

1. function level granularity , each endpoint is mapped to a function / method
2. protocol buffers : defines input and output schemas with backward compatibility resources ,
3. protobuf : its a compact data-type to share the data over the network in comparison to verbose text based formats like JSON , this improves speed of sharing , reduces     


* IDL : Interface description language, is a generic term for a language that lets a program or object written in one language communicate with another program written in an unknown language.


`statically typed language` : where the type is defined by the developer, at compile time and not at the runtime
`dynamically typed language` : where the type is defined by the program at the runtime



```
— HTTP connection — 

@app.route(‘/users/<int:user_id>’): 
def get_user(user_id): 
	return jsonify({ 
		‘user_id’ : user_id, 
		‘name’: 'Alice'
	})



call it using 

>> requests.get(“http://localhost:5000/users/12”)


-- grpc protocol  -- 

``user.proto``

syntax = ‘proto3'; 

service UserService  { 
	rpc GetUser (UserRequest) returns (UserResponse);

}


message UserRequest { 
	int32 user_id = 1;
}
 

message UserResponse { 
	int32 user_id = 1;
	string name = 2;
}
— 


`grpc code`

`python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. user.proto`

`
.protoc = proto compiler 
python_out = generates standard python classes from .proto definitions 
grpc_python_out = generate grpc specific python code ( stubs ) 
`

user.proto : using the above python command
user_pb2.py  : this file gets translated to `{file_name}_pb2`, this contains the actual `message`
user_pb2_grpc.py  : this file means `method` specified

```




