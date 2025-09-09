---
layout : post
date : 2025-02-15
title : Learning D3 js 
---

# Learning D3 js 
D3.js is declarative rather than imperative, meaning you describe what you want the visualization to look like, and D3 handles the details. For someone used to more imperative approaches (like traditional programming), this might feel a bit abstract or confusing at first.

## Simulation : 

Its like a mini physics engine, where instead of manually positioning every elements you let the simulation apply forces ( like gravity , repulsion , attraction ) to your nodes. 
f

Force Link : 

Force Many Body : Simulates repulsion between nodes 


## Smooth Transition 


## Register DOM event

Registers an event on DOM 

## Animation and tick in d3.js 

Its a part of d3-force, which uses a physics model to animate elements over time over time 
`tick` refers to each iteration of this simulation 

force simulation: it continously updates node positions based on forces ( like gravity , collision ) 
Runs in tick until stabilization or a specified limit 

If you manually modify the node position in the DOM, D3 simulation will override those values in the next tick !!
The simulation updates the values of x, y properties of each node based on forces. 

`fx` : fixed-x
`fy` : fixed-y

`<g> tag` : group tag 
`<svg> tag` : svg ( scale vector graphics ) 

Inside the svg we have g tags that have circle and lines inside it !! 

Dont change to manipulate things in DOM when in simulation mode as a resimulation will cause to redo the effects  

`force Simulation` : Simulation that acts / tells all the forces that are acting in a body, 

Repulsion : ( many - body forces ) prevents node from getting too close 
Attraction : ( link-force)  Acts like a spring between connected nodes, keeping related nodes together.
Centering force: Draws nodes toward a central point.
Collision force: Ensures that nodes maintain a certain distance from each other.


D3 PROPERTIES / METHODS TO USE : https://chatgpt.com/share/67b0f299-5328-8009-a2bc-4227d92681a8

Adding a class: 
`.attr("class" , "node")` : adds a class node

Access it using 

d3.select('.node') : pass the node name here !! 



D3 Code 
```javascript
const node = svg.append('g')  
    .selectAll('g')  
    .data(nodes)  
    .join('g')  
    .attr("class", "node")  
    .call(d3.drag()  
        .on('start', dragstarted)  
        .on('drag', dragged)  
        .on('end', dragended));  
```

Line-wise explanation: 

1) svg.append('g')
What it does:
    Adds a <g> (group) element to the svg.
    This <g> will contain all the nodes.

Why we do it:
    Helps organize the nodes in one group (useful for transforms, scaling, etc.).

2) .selectAll('g')

What it does:
    Selects all <g> elements inside the newly created <g>.
    Right now, there are no <g> elements, so this returns an empty selection.

3) .data(nodes)

What it does:
    Binds the nodes array (data) to the selection.
    Each item in nodes will be linked to a <g> element.

4) .join('g')

What it does:
    Checks if <g> elements exist for each data item:
        If they don’t exist, it creates a new <g> for each data item.
        If they do exist, it reuses them.
    Essentially, this creates <g> elements for every node in nodes.

5) .attr("class", "node")

What it does:
    Adds the class "node" to each <g> element.
    Now, d3.selectAll('.node') will correctly select all the node groups.

6) .call(d3.drag()... )

What it does:
    Adds drag behavior to each node using d3.drag().
    The .call() function applies this behavior to each <g> element.


## Add event listener to nodes !!

We can add event listeners using the 'click' functionality , 


```javascript 
d3.select("node").on('click' , (event , single_node) => {

    event.stopPropagation(); // this makes that event is restricted to here only and doesnt propagates furtehr  
})
```

Event listeners on elements are executed before event listeners on its parent elements.


## Add transition to the nodes !! 


## Understanding propery of a D3 Node 
[Important link](https://d3js.org/d3-force/simulation#simulation_nodes)

index - the node’s zero-based index into nodes
x - the node’s current x-position
y - the node’s current y-position
vx - the node’s current x-velocity
vy - the node’s current y-velocity

The position ⟨x,y⟩ and velocity ⟨vx,vy⟩ may be subsequently modified by forces and by the simulation. If either vx or vy is NaN, the velocity is initialized to ⟨0,0⟩. If either x or y is NaN, the position is initialized in a phyllotaxis arrangement, so chosen to ensure a deterministic, uniform distribution.

To fix a node in a given position, you may specify two additional properties:

fx - the node’s fixed x-position
fy - the node’s fixed y-position

At the end of each tick, after the application of any forces, a node with a defined node.fx has node.x reset to this value and node.vx set to zero; likewise, a node with a defined node.fy has node.y reset to this value and node.vy set to zero. To unfix a node that was previously fixed, set node.fx and node.fy to null, or delete these properties.

## Relevant forces

`.forceSimulation` : This gives few default things to the array !
they are : 
Alpha: Starts at 1
Alpha Decay: ~0.0228
Alpha Minimum: 0.001
Velocity Decay: 0.4
Required : This is required as the first step ( that makes the simulation to work ) it initialises a simulation on the nodes specified in it as an array 

```javascript 
const simulation  = d3
.forceSimulation(reactFlowNodes)

```

Named forces : These are the named forces, that means we have to name the force that we are using then we can manipulate the properties directly from it 

```javascript 
.force("charge" , d3.forceManyBody().strength(-100)) 
.force("charge").strength(-100);

// Here how we updated those forces, we name the params / properties of the force using its "name"  
```

Minimum distance below which the forces wont interfere  
Maximum distance over which the forces wont interfere
So if we are in a range that is between the min and max distance only then forces will appear 


`.forceX` : Directs the forces for nodes in x-direction 
So it says , hey all nodes you have a force applied in the x-direction and the coordinate is (d3.forceX(width/2)) or ( d3.forceX(50) ) 
and then we can set strength to this https://d3js.org/d3-force/position 


`.forceY` : Directs the forces for nodes in y-direction 
So it says , hey all nodes , you have a force applied in the y-direction and the coordinate is (d3.forceY(height/2)) or (d3.forceY(50))
so go to that force 

If applied in both direction we have a resultant force direction !


`.forceCenter` : Brings node to a center point, else if we have those unbounded ( no radius ) means they will be flowing any where onto the screen 

`.forceCollide` : This is a collision force that acts between bodies this is how the bodies gets seperated. each nodes it considered as a circle with radius. So a node is a circle that has a center and a radius property, the radius is the radius of the node so that is very small 


`.forceLink` : This force makes sure that if a single connected node in moved rest of the connected nodes along with it also moves in the given direction .. and we can defined teh 
Iterations are like the number of times you adjust the positions of the magnets (nodes) to make sure they are in the right place according to the forces (links) acting on them.
Distance : This is the distance between the edges 

`.forceManyBodies` : attractive or repulsive forces between bodies 

`.tick()` : tick, that is , in 1 second how many times will we recalculate the x,y,vx,vy for a node !! by default we have 300 times, so that is in 1 sec we update the simulation by 300 times !!
Iterations = we can increase of decrease it , its just a multiplication factor, (300 * iteration) this much time to do in a second 



## Params

`.alphaTarget()` : The simulation will go on and on till we reach this alphaTarget value

`.alphaDecay()` : The decay in which this happens, the alpha initial value will reduce in the exponential value in the decay value mentioned ! 

`.velocityDecay()` : Its the friction between nodes, it leads to stopping of 



## Using D3 with Reactflow 

Inside reactflow layout d3 action / params doesnt work , so we have to make a little trick here that is we have to get the nodes from original reactflow then pass them to d3 ( so that its forces can act on then ) 

D3 being very good only returns the vx and vy , that is , it calculates all the acting forces using the logic and returns the vx and vy along with updated x and y ... and that is done 300 times a second so we see it like a smooth transition as no major changes are happening !

Once we get the values from the D3, we update our reactflow nodes so that they are able to better interact with the whole react ecosystem and this happens in a use state over 300+ times in a second !! 

So this is a way to use reactflow in react ! 







