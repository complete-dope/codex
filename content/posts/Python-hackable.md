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
