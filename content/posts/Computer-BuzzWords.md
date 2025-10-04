---
layout : post
title : Computer buzzwords
date : 2025-01-30
---

This includes all cloud / devops , web-based , backend , machines etc 

# Devops Glossary


## Elastic Search 
"Elastic" – It is highly scalable and can dynamically handle large volumes of data.
"Search" – It is designed primarily for search and analytics, making retrieval fast and efficient.

Elasticsearch is a distributed search and analytics engine built on Apache Lucene. It is designed for fast, scalable full-text search and is commonly used for log analysis, real-time data indexing, and search applications.

Its a search engine with some database like properties, so its optimised for search , stores data in a json format and is used for logging purposes doesnt ensure ACID compliancy .Real time logging ( ~1s delay ) and regularly updates database. 


## Kibana 
Kibana is like a super cool dashboard for your data, especially when you're using Elasticsearch. Let’s say you have a lot of information, like numbers, text, or logs. Kibana lets you look at and understand this data in a visual way. Instead of reading through long lists, you can see the data in charts, graphs, maps, and other visuals, making it much easier to understand and explore.

It connects to Elasticsearch to pull out the data and then turns it into neat pictures, like bar charts or pie charts, that you can use to make decisions or find patterns.

Its just a UI tool 


## Grafana 

Grafana is a more general-purpose visualization tool that focuses on monitoring and time-series data (like server metrics, application performance, etc.) and can pull data from multiple sources like Prometheus, InfluxDB, and Elasticsearch. Grafana shines with metrics, time-series data, and real-time monitoring (e.g., CPU usage, memory usage, or network traffic). Grafana can connect to many different data sources, including Prometheus, Elasticsearch, InfluxDB, MySQL, PostgreSQL, and more.


## Prometheus

Prometheus is an open-source system designed for monitoring and alerting, particularly for time-series data. It is commonly used to monitor servers, applications, and infrastructure to track metrics like CPU usage, memory usage, disk space, and network traffic in real-time.

How It Works:

*Exporter: A software that runs on your system (like node_exporter for Linux servers) and exposes your system's metrics.

*Prometheus Server: Scrapes data from exporters at regular intervals and stores it.

*Alertmanager: If the data exceeds thresholds defined in Prometheus, it triggers alerts to be sent via email, Slack, etc.

Uses PromQL, its a query language, 


# DevOps Architecture Flow
k8s : kubernetes  
k9s : cli for k8s

<img width="1752" height="952" alt="image" src="https://github.com/user-attachments/assets/add1c3d0-eb82-44f3-9ee8-08086b646cd7" />

 

# Backend Glossary 

## Restful API 

A RESTful API (Representational State Transfer API) is a set of rules for building web services that allow communication between different systems over the internet using HTTP requests.


## CI/CD 
Continuous Integration (CI): A practice where developers frequently merge their code into a central repository, followed by automated testing to catch bugs early.
Continuous Deployment (CD): The practice of automatically deploying all code changes to production after they pass automated tests, ensuring rapid delivery of features and fixes.





# Machines 

## Virtualisation
Single host machine, but can serve multiple virtual machines from it .. Think of this as what google gcp does , its give you a single virtual machine inside a single host machine .. that means shared hardware resource but custom software / OS !!

OS that supports this is : proxmox , its an open source virtualisation OS, that lets you configure / setup your requirement using web interface and allots you a VM 

## Distributed inference 

Node : Single Machine
[Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

Multi nodes : Multiple machines





# Networking 

## Public IP Addresses
The IP address of your router that is actually connected to the internet and once u connect to a router you are basically in a private IP address range and its talk to internet using ports ...

Does public IP address also changes ? 
Yes, when electricity goes off , router gets offline , and when electricity comes back a new public IP address is assigned to the router unless you have a plan for Static IP address defualt is the dynamic IP address 

## Private IP addresses 
The router assigns you a local IP address and that can also change once you turn your wifi off and on. 

This typically starts from : 

* 10.0.0.0 to 10.255.255.255 
* 172.16.0.0 to 172.31.255.255
* 192.168.0.0 to 192.168.255.255

## Subnetting 
A subnet, or subnetwork, is a network inside a network. Subnets make networks more efficient. Through subnetting, network traffic can travel a shorter distance without passing through unnecessary routers to reach its destination.

A subnet mask is like an IP address, but for only internal usage within a network. Routers use subnet masks to route data packets to the right place. Subnet masks are not indicated within data packets traversing the Internet — those packets only indicate the destination IP address, which a router will match with a subnet.

![Subnet Image](https://cf-assets.www.cloudflare.com/slt3lc6tev37/2pBqIHUTSlxI7EW9XZPKf3/551ab3390ab9ab86fee15c73fd245f6c/subnet-diagram.svg)


# Proxy ( actual proxy ) 

This sits in front of client / user and makes his traffic seem to come from anonymous location and once it gets back the request it send it to the user without actually knowing the user that's actaully behind this !! 

A user can add proxy to anonymize himself / herself  

tools / examples : squid, privoxy, tinyproxy etc  

## Nginx / Reverse proxy

Lets say you have your python backend running at port 5000 and your node js backend running at port 5001 and your frontend application needs to sometimes call an python endpoint and sometimes an node JS endpoint .. so to maintain this from frontend woudl be a greater / tougher task so we use something like nginx / reverse proxy that acts a common route for all your frontend endpoint and then you can maintain it from there !! 
Incoming requests from users are first received by the reverse proxy, which then forwards those requests to the appropriate backend server to process and finally sends the response back to the client as if it came directly from the reverse proxy server itself, effectively hiding the location of your real web servers.

A server can add reverse proxy to do load balancing , 

It also does the same work of sitting in between and sending the request to relevant data site ... 

It also helps in load balancing , security of servers , caching also .. 


## Uvicorn 
Uvicorn is a high-performance ASGI (Asynchronous Server Gateway Interface) web server for Python. It's commonly used to run async web applications and APIs built with frameworks like FastAPI, Starlette, and other modern Python web frameworks.

In Uvicorn, "workers" refer to the number of separate processes that handle incoming requests in parallel. This is used when running Uvicorn with Gunicorn, which is a process manager that allows running multiple instances of Uvicorn.

Uvicorn is an ASGI server. It's responsible for handling the HTTP protocol, running your async event loop, and managing the actual processing of requests in each process. It implements the ASGI interface so that your application can handle async I/O operations efficiently.

Used for applications like FastAPI , Starlette 

## Gunicorn on uvicorn threads 

Gunicorn acts as a process manager. It doesn't handle the requests directly; instead, it spawns and manages multiple Uvicorn worker processes. In other words, Gunicorn distributes the incoming requests among these independently running Uvicorn processes, effectively scaling your application across multiple CPU cores.

Each worker is like a process and internally it uses multiprocessing on workers to spawn up each python process in a seperate thread so that is the reason behind that 

This is used for synchronous applications like Django, Flask etc

Gunicorn with uvicorn works as a process manager for managing the uvicorn ( async framework ) 

Each process is still uvicorn only , that is its async only so no need to use threading in that .. each process has its own single event loop and requests + tasks gets queued up in that event loop

## Web server 
A web server is essentially the "front door" of your application. It listens for HTTP requests—like when a browser asks for a webpage—and serves back static files (HTML, CSS, images) or forwards the request to something else if the content needs to be generated dynamically. Think of it as a very efficient file server that also knows how to handle simple routing tasks.


## Application server 
An application server, on the other hand, is where the heavy lifting happens. It runs your actual application code. When a request involves business logic—like querying a database, processing data, or interacting with other services—the application server takes over. It dynamically generates the response (like rendering a webpage or sending back JSON data) and then sends it back to the web server (or directly to the client in some setups).


How they all work together 

Nginx (front-facing web server)
Gunicorn (process manager)
Uvicorn workers (actual request handlers)

![Explained](https://pragnakalp.com/wp-content/uploads/2024/04/Untitled.png)




## PiHole  :
Acts as a DNS resolver, whenever you request for a domain, the DNS resolves matches that to an IP address, and then serves the request, and if you add Pihole at the front, it acts as a DNS resolver and it checks from it block list if that domain is mentioned in a blocklist networks and doesnt allow it to get passed !! 

This is amazing kinda like a custom DNS resolver !! 



# Low level 


## Memory leak :
When you dont unfree memory and it just piles up and RAM gets full and its leads to memory leak, global memory allocations are pilled up .. 
Garbage collector frees up memory that is unused in the code and is deferenced .. but any global variable is not picked up by garbage collected
so we can do :  `import tracemalloc` 








