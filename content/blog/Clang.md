---
draft: false
title: Starter's guide to C language and a bit of Machine learning in C lang 
date: 2024-12-22
---

C concepts

## LIBRARIES

`#include <stdio.h>` : standard input - output , printf , stdin , stdout 

`#include <string> ` : For string operations  string a = "this is a string"

`#include <stdlib.h> ` : (https://www.tutorialspoint.com/c_standard_library/stdlib_h.htm) standard libraries , that contains functions, size_t , free , allocate , malloc , realloc , exit , rand , srand , abort , exit , rand , RAND_MAX, EXIT_SUCCESS , EXIT_FAILURE

`#include <time.h> ` : time() , use 



`cos(double) , sin(double) , tan(double) , sqrt(double) , exp(double) .. ` : They all accept double's values

`cosf(float), sinf(float) , tanf(float) , sqrtf(float) , expf(float) .. ` : They all accept float values 



## PRINT STATEMENTS

```C
printf("The integer values is %d\n", integer_value);
printf("The float value is %f\n", float_value);
printf("The double value is %lf\n", double_value);
printf("The string value is %s\n", string_value);
printf("The character value is %c\n", character_value);
```



## C-UTILS

### STRUCT
```C
struct DataStrucuture{
    int age;
    string name;
    bool gender; 
} 
```


### MACROS
Macros in C are defined using the `#define` preprocessor directive. They allow you to create symbolic constants or functions that are replaced by their corresponding values or expressions during preprocessing.
Defines a constant value or a function that can be used multiple times



```C
#define PI 3.141592653589793238462643383279
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define NUM_TRAINING_EXAMPLES sizeof(training_examples) / sizeof(training_examples[0])
```


## C LANG. COMPILATION AND LINKING

This is a bit interesting part .. and the core of C lang .. 

`libm` : math library, used for advanced mathematical functions that provides mathematical functions like expf, sin, cos etc 
Written in C and some parts in assembly lang 

Compilation : Translates your C code into machine code for individual parts (produces .o object files).
Linking: Combines all the object files and libraries (like libm) into a final executable program.

The header files tells that compiler about declaration of these function but doesnt not include there actual code .. 

Linking tells the implementation of those functions 

`gcc gates.c -o gates -lm` : It does both compilation and linking 
The compilation part, `gcc gates.c -o gates` 
The linking part, ` -lm`  

First gets compiled then we do the linking so that it actually runs .. 


## Data Types in C

### Basic Types
- `int`
- `float`
- `double`
- `char`
- `void`

### Modifier Types
- `short`
- `long`
- `long long`
- `unsigned int`
- `unsigned char`
- `unsigned short`
- `unsigned long`
- `unsigned long long`

### Boolean Type
- `_Bool` (introduced in C99)

### Complex Numbers
- `_Complex` (introduced in C99)
- `_Imaginary` (introduced in C99)


## ARRAYS AND MATRICES 

### 1D ARRAY 
```
dtype array_name[size] = {value1, value2, value3, ...}

ex:
int arr[5] = {1, 2, 3, 4, 5};
float arr[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
bool arr[5] = {true, false, true, false, true};
```

### MATRIX


STATIC DECLARATION

```
dtype array_name[size1][size2] = {
    {value1, value2, value3}, 
    {value1, value2, value3}, 
    {value1, value2, value3,}
}


ex:

int arr[3][3] = {
    {1, 2, 3}, 
    {4, 5, 6}, 
    {7, 8, 9}
};


`second dimension needs to be passed (at least)`
float arr[][3] = {
    {1.1, 2.2, 3.3}, 
    {4.4, 5.5, 6.6}, 
    {7.7, 8.8, 9.9}
};


bool arr[][3] = {
    {true, false, true}, 
    {false, true, false}, 
    {true, false, true}
};

int [][2][3] = {
    {
        {1,1,1},
        {2,2,2}
    }
}

`While declaring a matrix, the first no. is not required to be explicitly mentioned rest no. needs to be mentioned`

```


DYNAMIC DECLARATION






## FUNCTIONS

```C 
dtype function_name(input_param1, input_param2 , ){
    // perform operations 

    return xx; 
}

void function_name(int a, int b){
    int c = a + b;
    // returns nothing
}

int function_name(int a, int b){
    int c = a*b;
    return c;
}

float function_name(float a, float b){
    float c = a*b;
    return c;
}

```





## FOR LOOP 


If the type is positive and unsigned_int then we can use `size_t` ::


```C
for (size_t i =0 ; i < count_size ; i++){   
        // perform operations    
}
```

<br>

If the type is `int` :: 

```C
for (int i = 0; i < count_size ; i++){
    // perform operations 
}
```




## WHILE LOOP


```C
while ( x < 10){
    // perform operation 
}
```


## POINTERS AND TYPEDEF 


### typedef
Alias to an existing data type , doesnt create a new data type rather create a new alias for existing data type  
`typedef existing_name new_name` 


Examples:
`typedef unsigned long ulong;` 
`typedef struct { int x; int y;} Point`
`typedef int* int_ptr;`
`typedef float array_of_3_floating_number[3]`


### pointer

Pointer to an integer, float ..  etc
`float* fp = 3.0f` 

Pointer to an float Array
`float* fptr = {1.0f, 2.0f, 3.0f}` 

Pointer to an 2D Array

```C
float fpmat[2][3] ={
    {1.f , 2.f, 3.f}, 
    {1.f , 2.f, 3.f}
};

float* mptr[3] = fpmat; // pointer to an array of 3 floats 
```



## BIT OF ML ALSO ADDED


![XOR , OR , AND](https://miro.medium.com/v2/resize:fit:1400/1*Tc8UgR_fjI_h0p3y4H9MwA.png)


### DERIVATIVE IN ML


Taking the derivative of the pure function vs function with values , does that make a difference ? let's say 

```python
USING FUNCTION WITH VALUES
dw1 = (x1*w1 + x1*eps + x2*w2) -  (x1*w1 + x2*w2) / eps ==> (x1*eps)/ eps = x1


// USING PURE FUNCTION
// dw1 = (x1*w1 + x2*w2) ==> dL/dw1 ==> x1

// Its the same thing, doesn't matter !!  
```

