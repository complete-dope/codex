---
draft: true
title: Learning Python things that I commonly forget   
date : 2024-06-09
tags: ["python","easy hacks"]
---

# Learnable hacks for python 🐍

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
