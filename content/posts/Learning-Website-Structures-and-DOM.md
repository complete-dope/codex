---
draft: false
date: 2025-01-16
title : Learning how web works and building my custom DOM for it !! 
---

[DOM Introduction](https://www.w3.org/TR/REC-DOM-Level-1/introduction.html)
[Browser Use](https://github.com/browser-use/browser-use)


# DOM
#### Understanding this line `The Document Object Model (DOM) is an application programming interface (API) for HTML and XML documents. `

DOM ( Document Object Model )
The DOM is like a map of a webpage or document. Imagine a webpage (HTML) as a tree where each part—like headings, paragraphs, buttons, or images—is a branch or a leaf. The DOM is the structure of this tree that makes it easy for programs to understand and interact with the webpage.

![Dom Tree](https://www.w3.org/TR/REC-DOM-Level-1/images/table.gif)

API 
An API is code that allows a program to "talk to" something else. In this case, the DOM API provides tools for programmers to interact with a webpage's structure.

Understanding it all, 
DOM is a "map" of a webpage or document. It provides a way (using an API) for programmers to interact with and manipulate the webpage's structure, content, and style.


Before the DOM, different browsers handled webpages in slightly different ways, making it hard to write programs that worked everywhere.

[DOM customs web browser's , mock web engines must be based on](https://www.w3.org/TR/REC-DOM-Level-1/level-one-core.html)
[Custom DOM implementation](https://github.com/browser-use/browser-use/blob/main/browser_use/dom/buildDomTree.js)

## What is xpath ? (XML path language) 
The absolute path of the node / which element to interact with , the absolute path of the node 

Example include: 

```xpath
html/body/div[3]
html/body/c-wiz[2]/div/div[2]/c-wiz/div/c-wiz/c-data
```

These start from the root html , till the absolute end of the node name 

The DOM is created by browsers , browser have a standard on how to create a DOM , the xpath is a way to interact with the DOM in an efficient way  

How to extract xpath ?



## Custom DOM 

Features of using a custom DOM , 
add features like is_clickable , is_visible etc .. 

## Making the bounding boxes:
The JS already gives / makes the bounding box for its elements and its very simple to get the coords of that bounding box. 


```javascript
const element = document.querySelector("#myElement"); // Select the element
const rect = element.getBoundingClientRect(); 

console.log({
    x: rect.x,           // X coordinate
    y: rect.y,           // Y coordinate
    width: rect.width,   // Width of the element
    height: rect.height, // Height of the element
    top: rect.top,       // Distance from top of viewport
    right: rect.right,   // Distance from right of viewport
    bottom: rect.bottom, // Distance from bottom of viewport
    left: rect.left      // Distance from left of viewport
});
```

CSS selector: 

```javascript

var idx = document.querySelector("div.container > p:first-child");
// This selects the para tag that has an direct first child present in the div tag that has an class of container  
```


To extract a css selector we can pass in xpath and get out a css selector 
that is : 

```javascript

function xpathToCss(xpath) {
    return xpath
        .replace(/\/\//g, '') // Remove double slashes
        .replace(/\//g, ' > ') // Replace single slashes with '>'
        .replace(/\[(\d+)\]/g, ':nth-child($1)') // Convert `[n]` to `:nth-child(n)`
        .replace(/@([\w-]+)/g, '[$1]') // Convert `@attr` to `[attr]`
        .replace(/^\s*>/, ''); // Remove leading '>'
}

// Example Usage
let xpath = "/html/body/div[1]/ul/li[2]/a";
console.log(xpathToCss(xpath));

```

"/html/body/div[1]/ul/li[2]/a"   to   "html > body > div:nth-child(1) > ul > li:nth-child(2) > a"




## The flow goes like

1. Creates all JS interactable tags from the custom DOM. 
2. Pass to the LLM model, and ask it to define the next step / where to click from there
3. Make bounding boxes using the core Algo, and then add labels to that , the labels should be same as what you got from the custom DOM parsing
4. Pass that to a good VLM model and ask it to find the changes it did ..       








