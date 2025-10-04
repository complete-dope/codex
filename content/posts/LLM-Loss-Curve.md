---
title : LLM's loss curve and compute 
date : 2025-08-16
layout : post 
---

## Small LM vs Large LM 

Small no. of parameter model has to choose between what knowledge to keep in parameter space and what to ignore and due to small no. of params the model tends to ignore most of the ood knowledge and keep the one that is commonly occuring ( small fields are ignored / tail knowledge is ignored ) 

### Multi-task learning : 
1. grammar
2. maths 
3. punctuation

.. etc 

### Heuristics : 
Small LM unwilling restricts itself to a smaller set of tasks , it can only improve a particular set / learn a particular subsets ( grammar , punctuation only ) while a large language model can learn both about tail knowledge / tasks like maths, punctuation and world knowledge 


### Whole loss 
The whole loss is the weighted loss of all the tasks LM's can perform and then it gives out a value 

<img width="661" height="476" alt="Screenshot 2025-08-16 at 4 27 02 PM" src="https://github.com/user-attachments/assets/63aff565-c7f1-4ad3-b826-709cd72cf1c7" />


### BIG-bench benchmark (containing 202 tasks)
This image shows the loss behaviour , how the losses on the tasks actually converge down !! 

<img width="955" height="548" alt="Screenshot 2025-08-16 at 4 33 51 PM" src="https://github.com/user-attachments/assets/044a3d1b-eaa2-435c-9022-01dc6f09d177" />

### Inverse Scaling / U shaped loss

Like a prompt like : "repeat after me , all the gliters is not glib , all that gliters is not __ " ,

Extra-small model is 100% correct on this 
Small model is 60% correct on this  
Large model is again 100% correct on this task 

The above task can be divided into 3 subtasks , repeat something , fix a quote , follow instruction and the below picture explain which model follows about this about the Language model

<img width="1337" height="424" alt="Screenshot 2025-08-16 at 4 41 55 PM" src="https://github.com/user-attachments/assets/75edcfbf-6c92-4e46-b731-839ec0ac5f3d" />



### LM's Research
Plot the scaling curve to actually understand how to further what works in a model and how to increase it further !!

this is actually a better / best way to understand the LM research 

<img width="661" height="429" alt="Screenshot 2025-08-16 at 5 00 47 PM" src="https://github.com/user-attachments/assets/87f9bc39-3efd-4ffe-bffb-b443e596aaa4" />



