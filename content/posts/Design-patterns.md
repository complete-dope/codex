---
title: Learning Design Patterns
date: 2025-10-15
draft: false
tags: ['design-pattern', 'python-learning', 'human-preferences', 'oops', 'interview']
---

# Python oops
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
2. Abstraction : Hiding away the complexities of the code so the code looks a bit clean  
3. Inheritance : Inheriting from a base class to avoid rewriting same methods  
4. Polymorphism ( many-forms) : so more like an abstract class , where define @abstractmethod and the class inheriting it has to use it   


# Design patterns 

1. `Strategy Pattern` : Think of this as a game strategy , the underlying input that is the game that is same for all strategies it's just that the implementation is changed so liek that the design pattern that we can use for all sorting functions

2. `Iterator Pattern `: dont tell the underlying Data structure like tree , linked-list , etc just use an `__iter__` and `__next__` method to sort this out.

3. `Adapter Pattern` : So this is bridge between 2 code base , the client wants in a specific format and you have implemented in other format so its a kind of adaptation from old code to a new code

4. 


## Decorators
<details>
  <summary>Python inbuilt decorators</summary>
  1. staticmethod : methods that doesnt access instance (self) or class data (cls) 
  2. classmethod : method that takes class 'cls' as first argument 
  3. functools : these are tools defined over class methods 
  4. dataclass : classes whose primary role is to store data those are dataclasses, and the variables defined here are still instance variables
</details>


















