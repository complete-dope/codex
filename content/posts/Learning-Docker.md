---
draft: false
title: Learning Docker
date : 2024-09-07
---

## Docker Commands end to end 

* **docker run** - Run a container from an image
* **docker start** - Start a container that has been stopped
* **docker stop** - Stop a running container
* **docker kill** - Kill a running container
* **docker rm** - Remove a container    

## Dockerfile 
`CMD ["/app/start.sh"]` :  This CMD command runs as soon as you run the container ( not at the build time ) , only at the run time this command works and easily gets overwritten by something like this `docker run -it <image_name> ./bin/bash` now bin/bash will run as the entry point !! 
`` : 


## Building Image from a Dockerfile
Once you have created the Dockerfile, now you need to  

So this builds an Image named 'backend', and we need to specify the '.' at the end that tells it to search for Dockerfile from here !!
`docker build --tag "backend" --file Dockerfile .`
This takes some time to build the image 

Once done, now check for 
`docker images` : This looks up for the docker images created from the last command    

### Docker in interactive mode

Here this `-i` is the interactive mode but this looks like shit !! 
```
docker run -i image_name /bin/bash
```

So we need to attach the -t ( tty ) to it to look good and usable
```
docker run -it image_name -p <host_port>:<container_port> /bin/bash 
```


### Docker in detached mode

```bash
docker run -d image_name
```

## After docker container is up and running 
`docker exec -it ed4f1a1d1880 /bin/bash` : Opens up a new terminal inside that container to interact with the processes running in the container. 























