---
draft: true
title: Learning Python things that I commonly forget   
date : 2024-06-09
tags: ["python","easy hacks"]
---

# Learnable hacks for python 🐍


## Python oops
Everything in python is an object (So a method with its variables / properties) that is also a object  

OOPS is not that simple as we use in projects , its an interesting low-level discussion 

`Class` variables :
So these are the one that are same for all the instance of the class and are from same memory location so changing any one of these will lead to change of other as well.
like this: 
`class Node: value: int , next: None` 


`Instance` variables : 
So these are defined instance wise, and are localised to each instance and each instance has its own memory space for this 

All Instance-variables and class-variables have the same  


OOPS has 4 main pillars: 
1. Encapsulation : Adding data fields to a class and then using it as class variable   

```python

# while doing encapsulation you also have to think about SOLID principles also

# designing a food ordering 
# whatever we are defining inside an class that all crud functions should also reside in that class only
class DeliveryFleet:
    name : 
    age: 
    availability : 
        
    def change_availability():
    def delivery_partner():
    def add_delivery_partner():

# designing a parking lot
class ParkingSpot:
    def can_fit(self, vehicle):
        pass

    def park(self, vehicle):
        pass
    
    def remove_vehicles(self):
        pass

class Vehicle:
    def can_fit_in(self, spot):
        pass
        
# Car class     
class Car(Vehicle):
    def can_fit_in(self, spot):
        return spot.size == Large || spot.size == Medium
    
class Bike(Vehicle):
    
class Truck(Vehicle):

# so here we have added can-fit in vehicle class cause tomorrow we need to add more vehicles that can be added without making any changes in the current fleet (so we that is not violating open-close priciple) 

```

2. Abstraction : Hiding away the complexities of the code so the code looks a bit clean


  
4. Inheritance : Inheriting from a base class to avoid rewriting same methods
5. Polymorphism : so more like an abstract class , where define @abstractmethod and the class inheriting it has to implement it or override it 

```python
class Agent:
  def __init__(self): pass

  def system_prompt(self):
    raise NotImplementedError()

  def say_hello(self):
    return "Hello"


class OpenAIAgent(Agent):
  def system_prompt(self):
    return "I am an open ai agent"

class GeminiAgent(Agent):
  def system_prompt(self):
    return 'I am an gemini agent'    

```
   

### `os library`

`os.path.abspath(__file__)` : this list the current path of file that we are on / scripting on   
`os.path.dirname(str)` : for whatever string path file defined we list the directory name for the same   
`os.path.getsize(path)` : outputs the byte count for that file    
`os.path.isfile(path)` : this outputs bool value of whether this is a True or False  
`os.path.isdir(path)` : directory path , outputs bool value whether this is true or false  
`os.path.exists(path)` : bool , tells if this path exists or not   

common error that are often repeated : 

<details>
  <summary>
    Relative Import with no parent package 
  </summary>

  `ImportError: attempted relative import with no known parent package`, the common cause of this error is Python sees file as a standalone script, not part of a package → relative imports fail. common fix is to run this as a module `python -m dir1.dir2.file` if this is your structure 

  Another method is to convert each folder to a package , that is by adding `__init__.py` here in each folder that way we can define each folder as a seperate package ... 
  
</details>

<details>
  <summary>ModuleNotFoundError: No module named 'file'</summary>
  convert that folder to a module by adding a `__init__.py` to that folder 
</details>


### Terminologies
* type annotations : variable that are defined with types like `counter:int`, this defines a way in which counter is an integer
* instance attribute : variable that are defined for each instance and are not just class defined ( in dataclasses attributes that are defined via annotations are in this manner )  
* class attribute : variable that are defined for each class
* 


### `sys library `



### python serializer

so python has an inbuilt serializer called `pickle` that is used to serialize (convert that to bytes) all *container dtypes* so what cannot be pickled are objects that contain things like : 
* open file handlers
* sockets
* lambda functions
* generators
* dict_values ( as that is not a type container )


Some more seralizers in python : 
1. json  : `import json`
2. pickle : `import pickle`
3. marshal : `import marshal`
etc (many open sourced also available)

network calls using API that use `json` as a serializer to serialize dataset into bytes and then that gets transferred over to TCP to make a network call 

`pickle` is designed for internal python processes only ,that is, if doing multiprocessing in python , or serializing something in python process to share to be consumed by some other python process, in these scenarios pickle is very useful in those scenarios  .. 

<details>
  <summary>
    TypeError: cannot pickle 'dict_values' object
  </summary>
  This here says that a you are trying to serialize dict values but as that is not a container dtype so that is not allowed in this case ... 
</details>



### Multiprocessing 
so multiprocessing in python depends on pickle library and we create processes using pool and inside pool we define max no. of processes that we need to run so this acts as 'semaphores' and this limits the compute ! 

Use `multiprocessing.pool(num_processes)` in python to distribute computation loads across, and this also gives a limit on no. of open processes at a time

So in a unix system, everything is a file and is read using its file offset and file descriptor so this becomes   
Each file read has 2 flags 
1. file-descriptor : OS level, this is how processes talk to each other 
2. file-cursor : tells what is open what is closed  


<details>
  <summary> 
    Runtime error : An attempt has been made to start a new process before the current process has finished its bootstrapping phase. 
  </summary>
That means you are trying to load something in your file that is recursively calling that function ( function that spawns more processes ) so that is the reason we use this `if __name__ == '__main__'` as a guard here , in macos this is how `spawn` method works 
</details>



### Guard 
if `__name__ == '__main__'` : This says if you are the main process running this then only run it if its importing from a subprocess ignore this part .. 

### functools
`partial` : this is to define the functions with some default paramter values  
`traceable` : 


### Network requests
To make mulitple network calls for a same session we can open up a client session that makes sure to keep a TCP /IP session alive for more time and helps between shared authentications between the sessions that can further help in lower latency (by avoid irrelevant handshakes) 

```python 
async with aiohttp.ClientSession() as session:
  async with session.get('') as response:


```

### Uploading a pypi package  

#### uv vs pip 
If your project setup is done using uv, then you need to do the `uv add <package>` to install the package in global such that it is reproducible .. 

Installing in editable mode :
<to check this out>

Intellisense : This completes the type check and type hints .. 
pylance : language server
pywrite : type hinter / writer
Editable install : this is done when you want to live debugs a python package and is installed using : `pip install -e . `  this just stores the pointer to the folder and we can edit the files and see the results getting updated at the same moment .. 

Always check `which python` , and also check this `which pip`, and these should point to same environments (not necessary as pip is a module and should be used via `python -m pip`   

Its common that sometimes pip doesnt get shipped in a python env so we can install it using `python -m ensurepip --upgrade` , this command installs the latest pip version and should always use the pip that is in the same env as the python and for doing that we can use it as a python module `python -m pip install <file>`    

* module : minimal code that is contained in a single .py file 
* package : python folder that is wrapped in `__init__.py` file 
* library : pyhton files that are related modules in a directory  
* framework : a comprehensive set of packages and modules that provides a complete structure for building an application.


  









