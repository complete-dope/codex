---
title: Learning Vue
draft: false 
date: 2025-01-06
---

Setup : 

* CDN -> script tag and get started ( no installation required )

* Vue CLI -> old not recommended way

* Create Vue -> use vite and ecosystem of plugins and is the recommended way  


# Vue.js
Written by someone that loves python 

Its a Progressive JavaScript Framework ( PJF )

Progressive JavaScript Framework, means even if your project is React you can use vue.js for some parts / components of it rather than fixing on vue from start .. nice benefit 


SPA ( single page application ) is a web application that loads a single HTML page and dynamically updates the page as the user interacts with it. 


Reactive data binding means a magical flow when the data updates the UI also updates itself , no need to manually write the updation logic ( done using virtual DOM ) 

Component based architecture , everything is a component , you can reuse it anywhere in your project


## Vue components 

The actual file that has .vue as the extension is called a component


Components has this following structures .. 
```js
<script>
    // JavaScript code - logic
    data(){

    }, 

    methods: {

    }

</script>

<template> 
    // HTML code - markup
    <div>
        <h1>Hello World</h1>  
    </div>
</template>

<style scoped>
    // CSS code - styles 
    // scoped means limited to the component
</style>

```

2 ways of building a component

1. Options API : old method 
2. Composition API : new method


We gonna start with the options API and then move to the composition API ..

### Vue Templates : 

`templates` are written in html + pythonic way = vue templates


* Conditionals In templates : 

`v-if , v-else-if , v-else`

```python
# In python we have  :

if age > 18:
    print("Adult")
elif age < 18:
    print("Not adult")
else:
    print("Age is not specified")
```


In Vue we do

```js
<script> 

<p v-if="age > 18">Adult</p>
<p v-else-if="age < 18">Not adult</p>
<p v-else>Age is not specified</p>

</script> 
```

* For loops :

`v-for`

```python 
for friend in friends:
    print(friend)
```

In Vue: 

```html

The key should be unique 

<template>
  <p v-for="friend in friends" :key="friend"> 
      {{ friend }}
  </p>
</template>
```

```html 

<!-- v-for also has index property and we can do it like  -->

<template> 
    <p v-for="(friend, index) in friends" :key="index"> 
        <span>
            {{ friend }}    
        </span>
        <button v-on:click="removeFriend(index)">x</button>
    </p>
</template>

```


Adding form:: 
`v-bind` : a 1 way binder, that updates only when data flows from parent to child flows
`v-model` : a 2 way binder, that updates both ways when data flows from parent to child and from child to parent

Acts as a default and binds to the value of input field here

```html 

<form v-if="present" v-on:submit.prevent="formSubmit">
    <label for="name">Enter Friend Name</label>
    <input type="text" id="fname" name="name" v-model="fname">
    <button v-on:click="addFriend">Add Friend</button>
</form>


```


### Options API : 
CODE IS IN : `App2.vue`

You define the data in the data(){//values} function, and if you want to call some methods / functions you can call it inside the methods dict

```js
<script>
 export default {
    data(){
        return{
            name: "John Due",
            present: false, 
            age: 20,
            friends: ["John", "Mary", "Tom"]
        }
    }, 
    
    methods :{
      toggleStatus(){
        if(this.present){
          this.present = false
        }
        else if(!this.present){
          this.present = true
        }
      }
    }
 }
</script>
```





### Composition API : 
CODE IS IN : `App_composition.vue`

The `template` remains the same, only the part inside the `script` tag is different and we make it cleaner 


Using `ref` to make it them reactive, and to use the variable defined in the ref we need to use the .value on them and then we need to do the things on that value ... 

Instead of data() and methods, here we use setup()


```js

<script>
  import { ref } from 'vue'

 export default {
  setup(){
    const name = ref("John Due")
    const present = ref(false)
    const age = ref(20)
    const friends = ref(["John", "Mary", "Tom"])
    
    const toggleStatus = () => {
      if(present.value){
        present.value = false
      }
      else if(!present.value){
        present.value = true
      }
    }

    return {
      name,
      present,
      age,
      friends,  
      toggleStatus
    }
  }
 }
</script>

```

A better way : 

Rather than defining the function setup() inside the script body, we can define it alongside script tag, this also removes the need to export and return things as that is added in `setup` added in the script tag

No indentation required for this 

```js

<script setup>
import { ref } from 'vue'
const name = ref("John Due")
const present = ref(false)
const age = ref(20)
const friends = ref(["John", "Mary", "Tom"])

const toggleStatus = () => {
  if(present.value){
    present.value = false
  }
  else if(!present.value){
    present.value = true
  }
}
</script>

```


### LIFECYCLE METHODS OF COMPONENTS  : 

* The lifecycle methods for components that are defined inside the `.vue` file for a single component only !!
Use JS map function to use an array of objects ( easily ) 

the `.value` extracts the initialised part of the internal dtype be it list/obj or something else

```js

onMounted(async() => {
  // code
})

```

## Vue Props

Props are used to pass data from parent to child components

The name of the prop can be anything, that u decide, 

```js
import myPropClass from '@/components/propclass.vue'; 

const Apples = ref("Bananas");
<myPropClass :title="Apples"/> //dynamic declaration , title =Bananas
<myPropClass title="Apples"/> // static declaration , title = Apples 

```


Same like we do in React:

```js

<script setup> 
import { defineProps } from 'vue';

defineProps({
  title: {
   type: String, 
   default: "Default Title"
  }
});

</script> 

<template> 
<p>{{title}}</p> 
</template> 

```

Even things in string in component / anything , none of that is actually string for static declaration for static need to pass that as name="Chatgpt" and is its dynamic / declarative then its :name="Chatgpt" 

Use the props as {{title}} in your template defined ! 


To bind anything to a dynamic value we can use , v-bind:prop="value" and to keep that static we can do , prop="value" 

:prop="value", to dynamically bind a prop to a value ( only use this to pass a static string )
:prop="10", an integer value of 10 bind a prop to a value ( the only way to pass integers / functions in Props )


## Vue Slots 
![Slots](https://claude.ai/chat/a2ffb7da-b4b9-439b-b0c0-045a168de154)

Slots vs Props in VUE ?? 

Props passes data to a function, they flow from Parent to Child

Slots are like holes in the child component that the parent can fill with content 

( bacho m voids hai and usko parents fill krte hai)


PROPS 
```js
<!-- Parent -->
<ChildComponent :message="hello" />

<!-- Child -->
<template>
  <div>{{ message }}</div>
</template>

<script>
export default {
  props: ['message']
}
</script>

```


SLOTS 

```js

<!-- Parent -->
<Card>
  <h1>Title</h1>
  <p>Content</p>
</Card>

<!-- Child (Card) -->
<template>
  <div class="card">
    <slot></slot>  <!-- Content gets injected here -->
  </div>
</template>

// the Title in H1 and Content in para gets added in the slot 

```


Named Slots / Holes 

```js
<!-- Parent -->
<Layout>
  <template #header>
    <h1>Header</h1>
  </template>
  
  <template #sidebar>
    <nav>Menu</nav>
  </template>
  
  <template #main>
    <p>Main content</p>
  </template>
</Layout>

<!-- Child (Layout) -->
<template>
  <div>
    <slot name="header"></slot>
    <slot name="sidebar"></slot>
    <slot name="main"></slot>
  </div>
</template>

```

To create a graph and as app is very dependent on this , then this all becomes very important to learn  ( learn vueflow library )




`ref` : can be used on any HTML tag, and acts as a reference to that can be accessed using $refs object. 

```html
<template> 
  <div ref="reference1">
    <!-- logic goes here  -->
  </div>
</template>
```


```html
<script>
const reference1 = ref(null); // initilized by a null value 


onMounted(()=>{
  if (reference1.value !== null) {

    // logic goes here 
  }
})

</script> 
```

Object Syntax, its a way to bind classes or styles to an element using an object. 

```html

<template> 

  <div id="temp-id" :class="{class-name : condition}" class="static-class">

</template>

```

Pairs are in Key : Value format, and only 2 places we can use the object syntax i.e. `:class` , `:style`  

But the values, can only be true-false 

If its true, then class-name is : `static-class class-name`
else the class name is : `static-class`




# Learning VUE FLOW (https://vueflow.dev/guide/ )

Install the @vue-flow from npm ... else specific pieces like @vue-flow/core , @vue-flow/node , @vue-flow/background , @vue-flow/controls , @vue-flow/minimap , @vue-flow/node-toolbar , @vue-flow/node-resizer

Background : 2 built in pattern for bg and that too is rest customisable 

MiniMap : bird eye view of the nodes in a small map at the bottomRight

Controls : bottomLeft zoom function , a control panel 

Node toolbar: essential tools directly from the Node itself ( cut, copy , paste for the Node button )

Node Resizer: Seamlessly adjust the size of the Node 


Starting with the coding part: 

npm add @vue-flow/core

Define the nodes and edges as they are the data part going then define them in the vueflow and pass them as Prop to VueFlow 

Nodes types: 

1. default ( can take input as well as output both )
2. input ( can only act as an input node for someone, connection dot at below )
3. output (can act as output node for some other node , connection dot at above) 
4. custom-type ( any other type , can have any name and is a custom node and can take a custom edge also !!) 

```js

const nodes = ref([
  {
    id: '1',
    type: 'special', // Here the type is special 
    position: { x: 500, y: 5 },
    data: { label: 'Node 1' , description: 'This is a node 1' }
  }
])



const edges = ref([
  { 
    id: 'e1->2',
    source: '1', 
    target: '2',
    type : 'special', // means this is a special edge 
    animated : true, 
    data: {
      conn: 'world',
    }
  }
])


```



data : any data/content you want to attach to a node or to a edge you can do that also and should be in a dict format

Node should have : 
id , type , position , data

Edge should have :
id , type, source , target , data


Then pass this data as a dynamic-props to the VueFlow component 

```html
<VueFlow :nodes="nodes" :edges="edges" > </VueFlow>
```

#### Slots in VueFlow: 

slot names start with `node-<type>` and `edge-<type>`

```html

<template> 

<VueFlow :nodes="nodes" :edges="edges" >

  <!-- the naming follows as node-<type> and edge-<type> -->
  <template #node-special="specialNodeProps">
    <SpecialNode v-bind="specialNodeProps" />
  </template>
  <template #edge-special="specialEdgeProps">
    <SpecialEdge v-bind="specialEdgeProps" />
  </template>
</VueFlow>

<template> 

```


#### CUSTOM NODES AND EDGES: 

For custom nodes you are GOD, you can do anything in it !! 


(Not good docs, need to experiment and learn it)

NODES : 

Choose a default node class ( vue-flow__node-default) and then add whatever you like in it !! 

Handle (connecting dots) : adds a connection point to the node, for custom node types you have to add it manually 

Position tells where to place that node 


EDGES : 
you have to define the bezier curve yourself


Theming , all class names  (https://vueflow.dev/guide/theming.html)


#### DEFAULT / INBUILT GRAPHS:

![Default Nodes Customisation](https://vueflow.dev/guide/node.html)

Handle means where the dot shows up in a node '.' , 4 positions possible , top , bottom , left , right

Set handle position using, targetPosition and sourcePosition


NODES: 

`data` : {validTarget : 'B', validSource : 'A'} ( that means A can only be connected to B and no one else )






EDGES:

`label` : "TEXT ON EDGE" , text to show on the edge

`markerEnd` : MarketType.ArrowClosed , the arrow at the end of the edge

`style` : { stroke: 'red' } , style of the edge [ color of edge ]

`labelBgStyle` : { fill: 'red' } , style of the label background

`lableBgPadding` : [8,4]

`lableBgBorderRadius` : 4

`labelBgStyle` : MarketType.Arrow

`updatable` : true ( play with nodes and make the updates )



Add Dark mode in graph : toggle dark mode in graph 



CONTROLS:
Add custom buttons to handle the graph better 


If button is visible but Icon is not visible that means, the CSS is broken and need to import / write some CSS for that particular class 



#### Advanced VueJS

`useRouter() hook` : It is used to create routes for the whole webpage.. basically useful for SPA ( single page applications ) the browser url changes and based on that we render a new component

The routes are usually defined in the `/src/router/index.ts`

```javascript

// router/index.js
const routes = [
  {
    path: '/dashboard',
    component: DashboardLayout, // 👈 This becomes `components.default`
    children: [
      {
        path: 'profile',
        component: UserProfile, // Child component
      },
    ],
  },
];

```


`computed` is used to create reactive computed properties. So whatever is inside the computed function will be re-evaluated when its dependencies change.
so when the route changes, Component automatically updates to reflect the new route's componet. Without computed(), it would just be a static value that never changes.

```javascript

const Component = computed(() => route.matched[0]?.components?.default)

```


## Listening to Events
`@keyup.enter` : listens for 'enter' button, means trigger this function when enter keyword is pressed    
:disabled="isLoading" : dynamic loading , if isLoading=True, then button is disabled , else visible

# LEARNING JAVASCRIPT

* Viewport : the display area of a screen, for a laptop its the laptop screen, for the mobile that would be the mobile phone screen 


*Math.random() returns a random number between 0 and 1

Path creates an actual path lines for the logo , svg 

Function declarations:


```js
function randomizeNodes(){
} 

const randomizeNodes = () => {
}

const randomizeNodes = (() => {

})

```

continue from here : https://claude.ai/chat/08b3a722-cb89-46d5-a84a-e517490163e5 

START WITH FINAL REWRITE OF eg THE CODE IN VUE JS 


onMounted() : Once all the components are displayed on the screen / DOM is updated . Then the functions inside onMounted() will get executed


Reactivity in Vue is amazing, the DOM is a connected graph of when an element flagged by `ref()` gets updated the DOM also gets updated and every component using that data also gets updated ( similar to when parent node gets updated the child nodes also get updated )   


Variables gets declared locally only in a JS function, and not globally ...   


# LEARNING CSS - TAILWIND CSS  

In CSS, the default logic is horizontal stacking of div tags, and to make that in vertical need to wrap that in a flex box / flex style. 



Using Tailwind css for styling .. 



`.basic-flow .vue-flow__controls .vue-flow__controls-button ` : 

What does this CSS line mean ? 
It means basic-flow has nested this class vue-flow__controls and that has nested vue-flow__controls-button inside it ... 


https://chat.deepseek.com/a/chat/s/27bc7208-1a9f-42a2-87b2-00b95b9df7dd 


`package.json`
It contains all the whole history of the project that you created, name, version, scripts-commands like (dev , build , ... etc ), dependencies, devDependencies that list all the packages that are getting used in this project. 
























