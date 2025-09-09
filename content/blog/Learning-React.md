---
draft: false
title: Learning basics of React  
date: 2024-12-04
---

[Doc link](https://docs.google.com/document/d/1Bg43BoC0xSmXugYI7L27NJm4ZFUaHDYlVYsFiTiodTM/edit?usp=sharing) 

[REPLIT](https://replit.com/@mohitdulani1/eg)

Trying to Learn React once again 
Delta to learn react is high, you need JS, css, html 

## `Importing / Exporting`
Named export ( export function App ) 
Default export ( export default <name> )

Default import ( import App from ‘./App’ )
Named import ( import {App} from ‘./App’ )  the { } must match the function name exactly 


Multiple Exports

```javascript

** File is  : app.js
export const Test1 = (()=>{

})

export const Test2 = function(){

}

export const Test3 = () => {

}


// call it as following

import {Test, Test2 , Test3} from './app.js'

```

## Components :

Different parts 
Like parts of a motorcycle , fan .. these are components and a site is made up many such components  
Component name should always start with a Capital letter
Inside JSX, we can write javascript also 
Can only return only one parent element 

Curly braces 
Inside curly braces, we can pass string , number and also for styling we can pass inside a js object inside `{}` so its `{}`

### Props
1. Passing to a jsx tag: Props are the information that you pass to a JSX tag
Passing to a component:

```javascript
<Component prop1 = {100} prop2 = {name:'mohit'}>
```

Receiving / Read props Way-1 :

```javascript
function Component ({ prop1 , prop2 }) { // prop1 and prop2  }
```

Object passing needs to be wrapped in double curly braces

2. Reading props Way-2 : function Component (props) read as props.prop1 and props.prop2

Child prop (passing other component as a prop, content inside a prop) : The content inside the element acts as a prop

Key prop (inbuilt keyword in prop) : used to identify one object from another 


### Template literals 
Passing in a combination of backticks + string + some dynamic value using $
** ` Hello ${Name} ` : Similar to python's F-string  


### Rendering
Going from code to beautiful frontend  using VDOM , virtual dom
DOM is a tree like hierarchy  
State changed ? First updates virtual dom 
Then uses diffing to find changes in real dom to virtual dom
Then updates few nodes in the real dom ( Reconciliation ) 

### Event Handling
[Event handling](https://youtu.be/wIyHSOugGGw)


Handling user interaction 
User clicking somewhere and the desired results is rendered using the event handler 
Event handler functions:
Are usually defined inside your components.
Have names that start with ‘handle’, followed by the name of the event.
Event handler props convention is to start with ‘on’ 

### Event propagation:
Each event clicks finds its location in a tree and from there it calls all the events above it to the top node of the tree. 
Stopping propagation, Event handlers receive an event object as their only argument. By convention, it’s usually called e, which stands for “event”. 
e.stopPropagation() → to stop up the propagation of e to the parent node 
Frequently used : onClick
onChange
onSubmit 


State
useState(initial_state)
useReducer() 

### Hooks
These can be called only at the top level of the component  
1. State Hooks
useState()
useReducer() 

2. Context Hooks
useContext()

3. Ref Hooks
useRef()

4. Effect Hooks
useEffect() : Runs once a component is rendered and you tell it to look for changes by providing dependencies. 

Setup code [ that performs something ] 
Use for everything that is non react like Calling an external API , external JS library , DOM event and we need to define when / how to call it , 



5. Performance Hooks ( not required in after react - 19 ) 
useMemo()
useCallback()

Purity
Same input should have same output and to always maintain this we can pass something called as <StrictMode>

Effects
Code that reaches outside react apps and calls other API’s

Context and useContext
Use context lets to read and subscribe to context created using create-Context
useContext >> is used when you dont want to pass on the data using props.. 
You can wrap the component inside the context provider and then use that component. 
   


Portals



Suspense 


Arrow function:
data.then((data) => console.log(data))
data.then((data) => {
   console.log(data)
})

Async arrow function 
data.then(async () => { })


Normal function:
function resolveIt(data){
   console.log(data)
}

data.then(resolveIt)


Fetching Data from API 
Using the fetch library / method from javascript 
But fetch is a asynchronous function and returns a promise you need to resolve the promise using .then() and .catch()

When you call a async function you gets a promise rather than the data … so you need to resolve that promise and if you need that API to be called and get resolved everytime to happen everything directly call it and resolve it ,.... 

 
UseEffect
For declaring / defining async functions, inside hook, the async function must be declared inside the scop and also called immediately ..

Do the desired roles what a function requires 
Then return a cleanup function 
Then [ ] to look for new changes 
 


==== 

Points to remember 

1_ The hooks should be declared inside the component as the components can be used at multiple places so making that global makes no sense .. 

2_ Props are read only, take the prop declare a state and then play with that state … its like this because of pure in react 

3_ react strictmode sometimes causes the component to render twice to check for its correctness .. this may be the reason of multiple renders that you see 



## useref hook 

It has a single object and doesnt causes re-render
Its used to reference / make reference to something, to access something we bind a reference to it (that we will use it)  
Its takes the DOM value directly

```javascript
import React, { useRef } from "react";

function App() {
  const inputRef = useRef(null);

  const handleFocus = () => {
    inputRef.current.focus();
  };

  return (
    <div>
      <input type="text" ref={inputRef} placeholder="Click the button to focus" />
      <button onClick={handleFocus}>Focus Input</button>
    </div>
  );
}

export default App;
```

The above function on clicking the button the focus will come to the input button 



## useEffect hook 

One of the methods for calling API but adds a conditional to when to call the API .. 

This is used to call API's  / external systems .. 
The thing is the function at it scratch implementation is a synchronous function and for it to work, even with fetch API we need to use useEffect() 

The basic fetching wont work ... 
useEffect takes in a function and a depedency array , that will call the function whenever something from the dependency array changes 

```javascript

useEffect(()=>{
   async function getData(){
      const response = await fetch('https://google.com')
      const data = await response.json()
      log(data) 
   }

},[dependency1 , dependency2])

```

This function runs whenever some dependency changes from the array, but can we pass params to the useEffect function ? 

Yup, the same old trick, if the function directly is unable to take anything wrap it up in another function and pass the props from that function to the useEffect hook ... 

Also used to call a function based on its state change, in the dependency array add all the states that need to tracked for and checked for change and useEffect hook just calls it ...   

There is also a cleanUp function, that resets the state back to normal to avoid irrelevant re-rendering ..   


###  React function component 

For react to know that its a React function component, it must start with a capital letter. 

```javascript

function getData(()=>{
   async function dataFetch(){
      const response = await fetch('https://localhost:3000')
   }
}, [])
// gives error like its not a react function component

// solve it by :

function GetData(()=>{
   async function dataFetch(){
      const response = await fetch('https://localhost:3000')
   }
}, [])

```


The onSubmit , onClick .. etc types of functions can take both sync and async functions as input !! --> kinda weird 


### Re-rendering 
Now that I added a new component ( that is just some html wrapped in JS shit for a bundler ) but its not getting compiled as react doesnt know that it got changed / it has to make changes because of some new addition ( fuck u react .. why can't you just listen to all the events occuring ) ... like an event loop 

So we have to now tell it that some new component has come please interact with it / show it to the user on screen  .. 


### Server side and client side component rendering !

So anything that a server sends ( means things that happen at the server and then sent back as a compiled HTML / JS)  is a server side code / component

And things that happen in client side ( like talking to an API ) is called Client side component ... 

There is one thing , we can't use async - await in client side code so how do we call API from client side ? 

Answer is : Using `useEffect` hook we can call the API from client side in a seperate component

The whole idea of components now makes sense .. without component writing things in react is just seems so impossible ..

### USING HOOKS

Using hooks makes sense, as the whole react code is async and for it to read values we need to use hooks , normal declaration of variable doesnt work correctly in this case  

### Component Rendering 

The way the component renders in React is quite interesting , at 2 times re-rendering happens, when a component's state or props change, the component re-renders.

If any state variable get defined inside a component, and it gets updated then the component re-renders and second is if the component takes a prop and that gets prop gets updated , the component re-renders



### Passing props to a button onClick function  
To pass props (or arguments) to a function in a button's onClick, you can use an inline arrow function that calls your function with those props. 


### Public Static 
Only the files in the /public/ are fetched by the initial HTTP request,  and the src/ requests are not sent .. By restricting static file serving to the public directory, Vercel ensures that sensitive source files (like configuration files, environment variables, or server-side code) are not accidentally exposed to the public internet.

Your source files in src are processed (bundled, minified, etc.) by the build tool. The resulting JavaScript, CSS, and other assets are output to the build directory (like .vercel/output/static or dist).

No Direct Access to src:
You cannot fetch or access files directly from src in production. They are either bundled into your app or ignored

### CORS Error 

Client to server , server sends backs data , browser sees if the header is set for `/same-origin/` , if yes, rendered else its CORS error 

Cors proxy in between , client to CP , CP sends request to Server , server sends backs data to CP,  CP adds 'same-origin' to it , CP sends its back to client , now that request comes from same origin !!   



























