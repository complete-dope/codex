---
layout : post
date : 2025-02-16
title : Learning nginx 
---

`Proxy` 
Something that sits in front of client is called proxy !

# HTTP and Frontend working 
So the frontend code renders on the user site (everyone knows but how does that happen ?)

The user requests the site url `https://mysite.com` , the http protocol goes to the hosted server and gets code for frontend site data and then render to the user system. In the frontend code we mention in depth what we want , how we want , how to interact with the data .. 

HTTP works on TCP / IP connection flow !


# Learning Nginx 

Nginx is a reverse proxy , that means is sits in front of the server and then its role is to redirect the traffic ( aka acts as a load balancer ) and hides the IP of the server 


## Setup and practices to perform 


Creates 2 directories 
```
mkdir -p /home/mohit/Desktop/mdesktop/reverse_proxy/sites-available
mkdir -p /home/mohit/Desktop/mdesktop/reverse_proxy/sites-enabled
```

sites-available : stores sites configurations
sites-enabled : contains symbolic link to configurations in sites-enabled


Now we need to add this in the nginx conf file as a directive..

Directive are the key values pairs in the nginx conf files like shown below:

```
include /home/mohit/Desktop/mdesktop/reverse_proxy/sites-enabled/*; 

include is the key
path is the value
These together are called directive 
```

Nginx configuration: 

```
server {
    listen 80;
    server_name mysite.local;

    location / {
        return 200 'Hello from your custom Nginx configuration!';
        add_header Content-Type text/plain;
    }
}
```

server : server block, that defines settings for a specific site 
listen 80 : tells nginx to listen on port 80 (HTTP)
server_name mysite.local : domain name this conf applies to  
location / { ... }: Begins a location block that handles requests for the root path (/) 
return 200 'Hello...';: Returns an HTTP 200 (OK) status with the specified text
add_header Content-Type text/plain;: Sets the Content-Type header to plain text


```
ln -s /home/mohit/Desktop/mdesktop/reverse_proxy/sites-available/mysite /home/mohit/Desktop/mdesktop/reverse_proxy/sites-enabled/
```

This creates a symbolic link to the configuration to the site ( CHECK : for location )  

```
chmod -R o+r /home/mohit/Desktop/mdesktop/reverse_proxy/
```


Then in the host file add the loopback IP address ( localhost ) to the domain name that we are assigning ( Sounds like DNS ?? ) 



Next steps : 

```
server {
    listen 80;
    server_name nodeapp.local;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

This sets up a reverse proxy for a node.js application running on port 3000 




## In More Depth:

Directives : key value pairs

```
proxy_pass  http://localhost:8000
```

```
events{

}
```
These are known as contexts, within context we have directive

## Writing a custom nginx file 

```
http{
  server{
    listen 8080;
    root /home/mohit/Desktop/mdesktop/reverse_proxy/
  }
}

events {}
```


Find the problem causing your nginx to not work : 
```
sudo journalctl -u nginx --no-pager --since "10 minutes ago"
```

Most common problem is access problem, u need to give access to all the parent folders and not just the final folder. 


## Mime Types

We need to tell nginx how to render which type of content .. that information is stored in mime types file and we just need to include it 

```
http  {
    include mime.types

    server {
        root /home/Desktop/<file-path>
        listen 8080;
    }

}

events {}
```

## Location blocks 


```
http  {
    include mime.types

    server {
        listen 8080;
        root /home/Desktop/<file-path>;

        location /fruits {
            root /home/Desktop/<file-path>; ( this auto adds /fruits at the end ) 

        }
    }

}

events {}
```

location /fruits : means this directory will only open when the endpoint is defined as in this case : http:localhost:8080/fruits 

using root : it appends the location defined in it at the end 
using alias : it doesnt append the location at the end so u can add at the end automatically 


By default, it looks for `index.html` files, so we can use try_files to try for these files also 

```
events {}

http { 

    include mime.types;
    
    server {
        listen 8080;
        root /home/mohit/Desktop/mdesktop/reverse_proxy;

        location /vegetable { 
            root /home/mohit/Desktop/mdesktop/reverse_proxy;
            try_files /vegetable/veggies.html /index.html =404; 

        }
    }
}
```

## nginx as a load balancer 

Lets say want to route it between 4 servers such that the time is saved per user .. 
what we can do is : 

create a dockerfile and expose the ports accordingly from it 

All the exposed ports becomes as a proxy_pass for a nginx file and we perform the routing from it !!

So what happens in a DIY home setup :

DNS routes the domain to IP 
Firewall passes that IP to the desired internal IP for which the port is exposed
The nginx should be running on that machine, and listening for requests on port 80 ( http ) or 443 ( https)

Once the request comes to that port , nginx then forwards that request to the internal IP's and (DOUBTFUL ON HOW?) get the data from there and sends back to the user ! 

[Nginx](https://www.xda-developers.com/how-to-set-up-nginx-reverse-proxies-in-your-home-lab/)

So we need a seperate dockerFile for frontend , a seperate docker file for backend and then nginx will need to be enabled from the frontend docker file and that will look for changes ( the init for nginx will be defined in the nginx.conf file !! )


nginx conf start -> looks for requests coming on port 80 


## Prod setup !!

First we create a frontend and backend dockerfile , and the nginx will be just the started as we do using systemctl and it will start listening to all the http ports 

In the nginx.conf file we will mention the port it has to redirect to which will be localhost:8000 based on the app

The app's port will be exposed using port forwarding in the docker file 

The backend will be similarly exposed using an endpoint running in the docker file !! 

So all we need is frontend docker rendering the sites and backend docker giving the response and nginx acting as the reverse proxy !!  




## Host the app on a VM 

Example scenario : 

Public IP  : 49.47.71.217
Private IP : 192.168.29.146

Hosted app on this private IP


all incoming for public router for http request from users should go to this IP address = 49.47.71.217:80 or 49.47.71.217:443 ( if https )


In router settings, all connections coming to the machine :80 , should go to this private IP : 192.168.29.146:80 and http should route that to the Hosted App , that routing will be done using nginx 


user will type http://publicIP:8080 , then directly it will see the app 

but the user typing http://publicIP , then also the user should be able to see and its enabled with using nginx


## Hosting multiple sites on a single server !!
TESTING HTTPS ON THE LOCAL SYSTEM WE CAN USE A SELF SIGNED CERTIFICATE , 

GET THE SELF SIGNED CERTIFICATE : `sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout server.key -out server.crt`

So the only limitation is the hardware , using a single machine we can host multiple sites from it and in the nginx itself we can mention to use which particular configuration for a single file !

For this we have to first simply run this: 

CONF FILE : `sudo nano /etc/nginx/sites-available/your-app` , This is the nginx conf file that we use

Then we need to make a symbolic link to the sites-enabled using this : 

SYMBOLIC LINK : `sudo ln -s /etc/nginx/sites-available/your-app /etc/nginx/sites-enabled/`
REMOVE THE SYMBOLIC LINK : `sudo rm /etc/nginx/sites-enabled/your-app`

RESTART : `sudo systemctl restart nginx`


``` 
server {
    listen 443 ssl;
    server_name 4.240.104.180;

    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket traffic
    location /ws/ {
        proxy_pass ws://0.0.0.0:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

}

server {
    listen 80;
    server_name 4.240.104.180;
    return 301 https://$host$request_uri;
}

```

## UPGRADING TO HTTPS AND NGINX CONF FILE 

1. SELF SIGNED CERTIFICATE 
Get a self signed certificate that lives for 365 days , but the signing authority cant be validated as you urself are trying to upgrade that connection ( ideally a 3rd party needs to do this part !!)

`openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365` : This saves the certificate at cert.pem file 


2. Free SSL Providers
Using providers like ZeroSSL and Let's Encrypt


ONCE GENERATED WE NEED TO TELL NGINX TO USE IT : 

here in the ssl-server we wrote the certificate name and stuff  

```
// binds to port 443 ( ssl / https )
server {
    listen 443 ssl;
    server_name 4.240.104.180;

    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket traffic 
    location /ws/ {
        proxy_pass http://0.0.0.0:8001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}

// binds to port 80 ( http )  
server {
    listen 80;
    server_name 4.240.104.180;
    return 301 https://$host$request_uri;
}
```



### GET AN SSL CERTIFICATE USING `NOSSL` 

So first step is you need to host a file that the ssl provider 

(you can host the file over http or https, for http no openssl certificate is required and for https you require to get a openssl certificate and use it !! )

then the provider tests for that file 

Once it verifies that this is the correct position for the file it upgrades your self signed (openssl) to the one provided by the authority !! 

Update the ssl certificate and the key so that you dont see that https error on it !!

DONE FOR HTTPS ! 

We need to do the same / similar for http but only change being we dont need to update and that will be the first time creation and updation !!
  
## common pitfalls 

Read the [vite docs throughly](https://vite.dev/config/server-options.html#server-hmr)  
vite server HMR ( hot module reload )
This fucks up in prod cases , so rather use 
```
npm run build
npx serve dist -l 8080
```

 
## What process happens internally ? 

The NGINX exposes port 443 

The internal whole setup runs on http port  

encrypted request comes from user browser > gets decrypted using nginx > internal routed to `http` as mentioned in the nginx conf file > uses this http for all the internal communication > before sending back the request encryptes it > sends back to user 

This process is called SSL termination - Nginx "terminates" the SSL connection and handles all encryption/decryption, so your internal services don't need to.

For WebSocket Traffic

Your frontend establishes a WebSocket connection to wss://4.240.104.180/ws/autosuggestion
Nginx identifies this as a WebSocket upgrade request based on headers
Nginx handles the SSL/TLS part (the "wss" protocol)
Nginx forwards the WebSocket connection to ws://0.0.0.0:8001/autosuggestion
Your backend server accepts the WebSocket connection
Two-way communication begins through this persistent connection

here is how nginx manages this magic connection part !!

`location /ws/ `: tells nginx to match any URL that begins with /ws/ > it founds this connection : `wss://4.240.104.180/ws/autosuggestion`
`proxy_pass http://0.0.0.0:8001/;` Forwards the request to http://0.0.0.0:8001/
`proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";` This is where the upgrade to websocket happens !! we append in the http request itself that this should be upgraded as a ws request !!

When your frontend code does new WebSocket('wss://4.240.104.180/ws/autosuggestion'):

The browser initiates a secure connection to your server on port 443
It sends an HTTP upgrade request with headers indicating it wants to switch to WebSocket protocol
Nginx decrypts this request using your SSL certificate
Nginx sees the /ws/ path and applies your WebSocket location block
Nginx forwards the WebSocket handshake to your backend at http://0.0.0.0:8001/autosuggestion
Your backend accepts the connection, and a full-duplex WebSocket channel is established
Nginx acts as a transparent proxy, passing messages back and forth between the browser and your backend

NGINX seamlessly handles the protocol conversion part !!
































 
