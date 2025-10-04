---
draft: false 
date: 2025-07-20
title: Building Backend Architecture from scratch 
tags : ["django", "Django ", "Backend", "microservices"]
---

# Building a scalable backend arch

## Using Django / django 

We start with a django Project, 

Using : `$ django-admin startproject <project-1> <dir_name>`


A single django project contains its own views, urls, asgi , wsgi , manage.py files and a project is often called as service so these 2 things projects and services are same only, and the microservices architecture is the one where each service aka project can be scaled up independently .. 

And each project can have multiple apps in it and we create an app using :
`python manage.py startapp <app_name>` this also initialises the models.py, apps.py , serializers.py , admin.py pages


so to create an new project using:  `django-admin startproject <project-2>` and similarly for multiple projects we do this same !
And likewise each project can have multiple apps in it ! 

```
Django_project_1/ 
    --- App1 
    --- App2 

Django_project_2/ 
    --- App11 
    --- App22 
```    

### ORM in Django ( object relation manager )
This is the auto-schema creation in django that is provided at the module level ... in the `models.py` that is present at app level, we define the schema and that is mapped to the ORM using the database tables and the migration to sql tables is done using the `python manage.py makemigrations`


Real time chat system : 

> Websockets
> Kafka messaging queue


### Communication stack 

> gRPC using proto-buffers, no REST 
> 


### Backend Stack 

> Golang
> 
