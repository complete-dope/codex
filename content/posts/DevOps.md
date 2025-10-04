---
date: 2025-08-31
title: Home Lab
layout : post
---

Tags : Home lab, k8s , k9s, k3s , kubernetes, jenkins, docker, helm, kind, kubectl


## Kubernetes 
This is a manager of docker containers 

So assume it like this , we have a VM from azure , that has some specs and we now want to spin up 4 containers in it so to do this we need manager as well that handles the routing, load balance, code updates reflection ( CI/CD ) and someone that we can talk to at manager level and it gives all information about the internal working of the containers, logs etc so that role is taken cared by kubernetes (k8s) 


## Docker Container 
Create your dockerfile , then create its image and then from image create a container, so this is the same that as before only additional step is to register this container to a registry ( docker hub / private hub like gitea)  

so pull the image from the hub and use it 


## Pull from registry
Now pushing to a public registry like docker-registry, makes your container public and anyone can see the code if not obfuscated .. 
So to avoid this public drama , we use private registry and github provides us that using its registry called GHCR (github container registry ) 

So in your workflow you have to create files in `.github/workflows/deploy.yaml` and in that yaml file you have to mention what needs to do be done so the bare minimum / basic part that we do / basic script that we do it 

This is a github action ( that auto publishes the latest code in ghcr ) 

```
name: Build and Push Docker Image

on:
  push:
    branches:
      - llm_service

jobs:
  build_and_publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - name: Build and push the image 
        run: |
          docker login --username complete-dope --password ${{secrets.GH_PAT}} ghcr.io
          docker build astrology_api/ --tag ghcr.io/complete-dope/astrology-api:latest
          docker push ghcr.io/complete-dope/astrology-api:latest

```
This says to run these commands as soon a push is register on a particular branch name .. 
and the password is stored in repo's secret access so take that up from there 

now the container is up there and running in the ghcr (github container registry)

in jenkins just pass this and pull it from there  

---  
Till here now we have k8s and docker container and the container that's pushed to a registry 
---

## Kubectl 

Its a tool to talk to the k8s manager that is managing my containers ... and its tells me information about them that is :


get default pods : `kubectl get pods` ( by default its only gives the default one )

get all pods : `kubectl get pods -A`

<img width="583" height="60" alt="Screenshot 2025-08-31 at 6 56 18 PM" src="https://github.com/user-attachments/assets/6ad44fc0-1c83-4082-ad49-1d910d4c32a3" />


services : these are the one that are exposed , the one through which we talk to, these are fixed IP and pods internal IP change but this doesnt change !! 

`kubectl get svc`

<img width="583" height="60" alt="Screenshot 2025-08-31 at 6 56 42 PM" src="https://github.com/user-attachments/assets/5f544151-8585-45aa-9dce-8c0a64561447" />

ClusterIP : k8s ip, this is the internal one to one communication not its not exposed 

NodePort : Port access from your laptop to the cluster


## Starting the k8s service 
install kubernetes, kubectl , KInD ( kubernetes inside docker ) 

This kind is used for development only ! 

Create a CLUSTER using kind (if not on a VM), using 
`kind create cluster --name dev-cluster`

check : `kubectl cluster-info`

To deploy anything inside k8s we need to write that out in a .yaml file and then run : `kubectl apply -f nginx-deploy.yaml` this is how to deploy and create a service in k8s ( given your yaml file mentions the deployment and service to register )

Similarly we need to get muliple common apps to use inside k8s to do that we can use helm that are like app store having all relevant `.yaml` files inside it and then use it to add to containers

## YAML files 



## Helm deployments
This is like the app-store for apps .. 
Lets suppose you want to do the jenkins integration for your CI/CD flow how will you do that ? 

get jenkins docker image, create yaml files ( 5-6 of them , deployments, services, RBAC, configmaps, secrets, persistent volumes, etc) and then apply to k8s using `kubectl apply -f k8s/deployment.yaml` and that is a very manual process to follow.   


Rather than creating multiple yaml file for each standard app to deploy we can use helm charts that directly install and add in the add and from there we use it in rest of the app  


## Jenkins

This is a CI/ CD framework that pulls from a repo , builds docker image from there and register that image to a hub registry and tell k8s to pull the new image from there and use it 

In jenkins we upload the yaml file and create a `pipeline` that tells from where we can   that jenkin s
We can create a jenkins file for the same as well  

Creating a jenkins pod on k8s using helm 

first fetch is from helm charts 
`helm repo add jenkins https://charts.jenkins.io`
`helm repo update`

Create a namespace 
`kubectl create namespace jenkins`

Install jenkins with helm 
`helm install jenkins jenkins/jenkins -n jenkins` 

This takes the default jenkins image you can customize it with your own image using custom yaml file 
`helm install jenkins jenkins/jenkins --namespace jenkins -f values.yaml`

See if jenkins is active using 
`kubectl get pods -n jenkins`

Then run the pod using this command : 
`kubectl port-forward svc/jenkins 8080:8080 -n jenkins`


## Prod flow 

In prod, create a cluster then add the jenkins to that and manage all from jenkings that's how it works in prod level !!  

Create multiple clusters and configure each individaully and then use a lb at front to expose them !! 
`kind create cluster --name <cluster-name>`



### delete a cluster

get clusters : `kubectl config get-clusters`
delete cluster : `kubectl config delete-cluster <cluster-name>`
























