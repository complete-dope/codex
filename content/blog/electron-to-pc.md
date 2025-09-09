---
layout : post
title : Learning computers from scratch 
date : 2025-01-11
---

Links : 
* http://www.gooath.org/blog/2022/12/05/how-do-computers-really-work-part-1-basics/

[Transistor](https://youtu.be/IcrBqCFLHIY?feature=shared)  
[Diode](https://youtu.be/Fwj_d3uO5g8?feature=shared)  
[Logic gates from transistor](https://www.youtube.com/watch?v=sTu3LwpF6XI)  

# Basics 

Atom made up of electron , proton , nucleus 

conductor -> free electrons in valence shell 
insulator -> no electrons in valence shell 
semiconductor -> few electrons in valence neither a good conductor or insulator 

semiconductor in pure form is useless , dope it to make n / p type 

## Diode
Electric field: start from positive end to negative end 
electrons accelerate in the opposite direction of the field 

n-type connected with p-type to create a np-junction for unidirectional flow of current   

A depletion zone is created at junction which has around 0.7v Potential diff and to flow the current a battery of >0.7 is required else its in off state 

To open-close the connection human involvement is required ( major drawback ) 

## Transistor

N-P-N type and P-N-P type 

In npn, 
np side and pn side both want to flow current in inner sides so no outer current is flown in natural state 

when p side is connected to a insulated-positive plate electrons starts to flow from n side to the p side and in addition with an external Electric field (using an battery) the current starts to flow in the required direction.  

Bit is a transistor, 8 bit combined make up a byte ( 8 tranisistors combined )  

Create logic gates from transistor using AND , OR gates ..etc 


To create a basic NAND operator, 2 input nand gate you need 4 transistors ... 

A single transistor can produce only a single bit output 

Combine logic gates to create Adder, Subtracter, multipliers, divider etc   

Adder, subtracter are used in computers to make the work happen 


The most important thing is Transistor powering all the devices in the world ... but the things come to how much can we compact it in a single board ?
There comes Moore law limitations 
The transistor cant be just made smaller as at such small levels, atomic forces comes into play and electrons start flowing from n-type to p-type without additional aid .. so that makes a transistor usesless .. 


# Advanced 

We created an Adder and a Subtracter in the section above this is used in the actual system to do the mathematical calculations   



## Electricity

Inverter : Converts to DC to AC power

Rectifier : Converts from AC to DC power 

Inside a copper wire, electrons flow randomly , we need to give them a direction that is done using voltage (DC) and electron tend to come back to their initial position so they move through the appliance / switch and light it up 





