---
draft: false
title: Learning TypeScript
date : 2025-02-28
---

[Typescript](https://www.youtube.com/watch?v=BCg4U1FzODs)

# Intro and Setup for Typescript

Superset of Javascript, its built on top of JS 

JS is a dynamically typed language, where we dont define the types and they are associated with run-time also like python
whereas,
TypeScript is a statically typed language, you define types to variable like , C , java 

You need to compile Typescript to work for your case, it compiles .ts to .js and   

## Learning typescript 

```javascript

let id = 5; // js
let id: number = 5; // typescript

```
Here we defined a type and as the modern browsers can only understand JS and no this fancy languages so we need to compile this to JS. 

To convert to JS we use `tsc`, that is typescript compiler 

`tsc index.ts` : This compiles the index.tsc file to index.js file 

Create at typescript config file, using `tsc --init` 

This creates a file named `tsconfig.json`, that contains all the confs for your .ts file, where you mention the root directory , the source directory , javascript version to compile to .. etc 

`tsc --watch` : watches for all tsc files and auto compiles them in the destination directory mentioned 



## Data-Types and Usage in Typescript 

```javascript

// All PRIMITIVE TYPES in typescript 

// string: Textual data (e.g., "hello").

// number: Numeric values (integer, float, hexadecimal, etc.).

// boolean: true or false.

// null: Explicit absence of a value.

// undefined: Variable not yet assigned.

// symbol: Unique, immutable identifier (e.g., Symbol('key')).

// bigint: Large integers (e.g., 100n).


// All SPECIAL TYPES in typescript

// any: Opt-out of type checking (use sparingly).

// unknown: Type-safe counterpart of any (requires type assertion before use).

// never: Represents unreachable code (e.g., function that always throws an error).

// void: Absence of a return value (common in functions, e.g., function log(): void).


let id: number = 5;
let company: string = "Traversy Media";
let isPublished: boolean = true;
let x: any = "Hello";
x = true;


let age: number
age = 50;


// Array of numbers
let ids: number[] = [1, 2, 3, 4, 5]; // ids should be a list of numbers / integers

// Array of any
let arr: any[] = [1, true, "Hello", 4, 5];


// Number , String , Boolean (tuple of 3 data types)
let user: [number, string, boolean] = [1, "Traversy", true];
let person: [number , string, boolean] = [1, "brad", true];


// Array of tuples of number and string   
let employee: [number, string][] 
employee = [
    [1, 'hello'],
    [2, 'world'],
]

// Union
let pid: string | number

pid = "12";
pid = 22;  

// Enum 
enum Direction1{
    Up, 
    Down, 
    Left, 
    Right
}

enum Direction2{
    Up = "Up", 
    Down = "Down", 
    Left = "Left", 
    Right = "Right"
}

// by default the values are 0, 1, 2, 3 in the order !! 

console.log(Direction1.Up); // 0
console.log(Direction1.Down); // 1
console.log(Direction1.Left); // 2


console.log(Direction2.Up); // Up
console.log(Direction2.Down); // Down
console.log(Direction2.Left); // Left

// Objects (key-value pair) (key-name : data-type )
type User = {
    id: number,
    name: string
}

const user2: User = {
    id: 1,
    name: "John"
}

```


## Bit complex in TypeScript

Type-Assertion : Its a way to tell the compiler, "trust me, I know what type this is"

Its a way to tell typescript, that u as developer know about its type than TypeScript's type checker .. 

```javascript
// Example-1
let value: any = "hello";
let strLength: number = (<string>value).length;

//Example-2
let value: any = "hello";
let strLength: number = (value as string).length;
```

Overriding implicit 'any' types

Interesting Example


```javascript
const inputElement = document.getElementById("email") as HTMLInputElement;
inputElement.value = "user@example.com"; // Now TypeScript knows it's an input
```

Possible types for document.getElementById("email") are either an HTMLElement or a null value

So a code like this :

```javascript
const inputElement = document.getElementById("email");
inputElement.value = "user@example.com"; // ❌ Error: Property 'value' does not exist on type 'HTMLElement'.
```


So we need to do a type assertion, so if I make that to HTMLInputElement, I will tell typescript explicitly that this is not any HTMLElement its a HTMLInputElement 

```javascript
// Suppose the element with ID "email" is a `<div>`:

<div id="email">Hey I am Email in a div tag</div>
const inputElement = document.getElementById("email") as HTMLInputElement; // ex-1
const inputElement = <HTMLInputElement>document.getElementById("email");  // ex-2

inputElement.value = "test"; // 💥 Runtime Error: `value` property doesn't exist on `<div>`.
```

You can't leave any variable untouched in typescript , the sole purpose of typescript is to be able to do better type checking. 

```javascript
function addNum(x: number, y: number): number{
    return x+y;
}


// This defines that output is also a number 
```

Either you have to define it as a any or void 


Complete typescript with comments ! 

```javascript
// Void

function log(message: string | number ): void {
    console.log(message);
}
```

//Interfaces 


```javascript


// Interface

interface UserInterface { // 
    id: number,
    name: string,
    age?: string // This is an Optional field   
    readonly email?: string // This is a Readonly field
} 

const user1: UserInterface = {
    id: 1,
    name: "John"
}


// interface inside function 


interface MathFunction {
    (x: number, y: number): number
}

const add2 : MathFunction = function(x: number, y: number): number {
    return x + y
};

const subtract: MathFunction = (x: number, y: number): number => x-y;


// Classes 


class Person{
    id: number
    name : string

    constructor(id: number, name: string){
        this.id = id;
        this.name = name;
        console.log("Hello World");
    }
}


const mohit = new Person(21 , "Mohit");
const john = new Person(22 , "John");


mohit.id = 123;

// Public, private, protected 

// protected: used inside the class and its subclasses
// This = self, in python  
class Person2{
    private id: number 
    public name: string 
    protected age: number


    constructor(id: number, name: string, age: number){
        this.id = id;
        this.name = name;
        this.age = age;
    }

    function(a: number, b: number): void{
        console.log(this.id);
        console.log(a+b);
        
    }
}


// interface in a class 

interface PersonInterface{
    id: number,
    name: string,
    register(): string
}

class Person3 implements PersonInterface{ // This si just an extra type-checking and does nothing interesting ... !!   
    id: number
    name: string

    constructor(id: number, name: string){
        this.id = id;
        this.name = name;
    }

    register(){
        return `${this.name} is now registered`;
    }
}

// subclass


class Names extends Person3{
    position : string 


    constructor(id: number, name: string, position: string){
        super(id, name); // Take the parent class implementation of constructor and use that ! 
        this.position = position;
    }
}

// Generics 
function getArray(items: any[]): any[]{
    return new Array().concat(items);
}

let numArr = getArray([1,2,3,4]);
let strArray = getArray(["Hello", "World"]);

numArr.push('Mohit'); // This gives no error as its of `any` type

// How to change this ? 

function getArray2<T>(items: T[]): T[]{
    return new Array().concat(items);
}


let numArr2 = getArray2<number>([1,2,3,4]);
let strArray2 = getArray2<string>(["Hello", "World"]);

// numArr2.push('Mohit'); // This gives an error as its of `number` type
numArr2.push(5); // This is fine as its of `number` type

```


[TypeScript for React](https://github.com/typescript-cheatsheets/react#reacttypescript-cheatsheets)

































