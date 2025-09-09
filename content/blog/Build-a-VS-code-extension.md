---
draft: false
date: 2025-01-18
title: Building a VS code extension
---

[starter](https://www.youtube.com/watch?v=q5V4T3o3CXE)
[basics of node and npm](https://youtu.be/P3aKRdUyr0s?)

# Basics of NPM and Node

NPM : Node package manager 
A collection of all the open source JS work that people from all over the world contribute to. It helps in simplifying your JS work as you dont have to write everything from scratch 

Nodejs : 
Javsascript runtime which does JIT compilation for JS , unlike python which uses cypthon for compiling code to C and some of it uses 

package.json : will list all the packages your project is using so if your project has package.json user can directly use it to install all of those packages 

# Building a VS code extension from scratch 

start with `npm install -g yo generator-code`

yo : yeoman , automates boiler plate project templates 
generator-code : to install vs code sample extensions using yo, so yo named this as generator-code

`extension.js` = This is the main entry point for your extension. Customize it to implement your extension's functionality.

It contains the code that runs when the extension gets activated inside a function called activate and runs at the first time ... 

Add the commands , add event listener , implement features like webview, status bar items etc 

`package.json` = dependencies , contribute commands ( which tells it where to route for a particular input ) 

## After all this initialisations what next to do ? 

## Webview
Everything that we see in a vs code extension is built out in a webview ( a single extension can only have a single webview ) , 
That webview is can be built in HTML , css , JS or using a fjramework like React, Angular, Vue JS .. 

In the webview initialise a react / typescript project and code it, the directory will be '/src/components/'  


## Interacting with webview 
For the webview to interact with the backend code, the vscode.postMessage with a type {"<>"} (posting the event), the event needs to be posted so that means we need to use switch-case for that to work ...

This way the webview / frontend posted a message : 

```javascript
vscode.postMessage({
  type: 'takeScreenshot',
  value: {provider, modelName , question}
});
```

To recieve that message in the backend / extension.js we have to use onDidReceiveMessage: 

```javascript
webviewView.webview.onDidReceiveMessage(data => {
  switch(data.type){
    case 'takeScreenshot':
      try{
        // js code that works
      }
      catch(err){
      }
  }
})

```



## BASIC FLOW FOR AN EXTENSION 

`package.json` : tell the command that you are going to develop

`extension.js` : register that same command and tell in the function what you want it to do 


## Package.json
The 5 most important section is contributes section ,: 

1. `commands` : That we use using ctrl+shift+P , that get registered there

2. `viewContainers` : Defines containers in the VS Code Activity Bar (the sidebar on the left). These containers hold views (e.g., the Explorer, Source Control, or your custom views).

```javascript
{
  ID: 
  Title: 
  Icon : 
}
```

3. `views` : Defines the actual content , inside that panel , view that will be shown in the UI
The id defined in the activity bar for an extension , for that only you can create a UI also , that means if inside viewsContainers, I defined an ICON with an extension as:
defined in the viewContainer :

{
Id :'ext_testing',
Title : 'Testing this extension ',
Icon : 'Images/icon.png' 
}

How to use this in views file : 

```javascript
"views": {
"testingext": [
  {
  "id": "testingext.view",  // this is view-ID
  }
]
}
```

4. `Menus`
The id for the menu it takes from the defined commands, first you need to register that command, then based on the ID you gave it , you need to call that ID and use it in the menus with the :
```javascript

"menus": {
  "view/title":[
    "command": <as defined in the commands>
    "when": <when to call it>
  ]
}

```



### DISPLAYING A WEBVIEW :
To display a webview, that means first I need to register a provider , using registerWebviewViewProvider, that takes the ID of the view-ID, and a custom web view , then resolve the web view and set options, and then set the view  


```javascript
// In the extension.js file 

class CustomWebView {
  // defined the extension URI for the model  
  constructor(extensionUri) {
    this._extensionUri = extensionUri;
  }

  resolveWebviewView(webviewView) {
    this._view = webviewView;

    // Set up the webview options
    webviewView.webview.options = {
      enableScripts: true, // Enable JavaScript if needed
      localResourceRoots: [this._extensionUri] // Allow loading local resources from the extension URI
    };

    // Set the HTML content for the webview
    webviewView.webview.html = this._getHtmlForWebview();
  }

  _getHtmlForWebview() {
    // Generate the HTML content for the webview
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <title>Custom WebView</title>
          <style>
              body {
                  font-family: Arial, sans-serif;
                  margin: 0;
                  padding: 0;
                  background-color:rgb(0, 0, 0);
              }
              h1 {
                  color: #333;
                  text-align: center;
                  margin-top: 20px;
              }
          </style>
      </head>
      <body>
          <h1>Hello, this is a Custom WebView!</h1>
          <h2>You can customize this content as needed.</h2>
      </body>
      </html>
    `;
  }
}


```

What if you want to use React ? 
The old js engines cant understand the modern es modules / syntax so we have to use something like bundler .. 

But, what is a bundler ?? 
Bundlers like webpack , ship your whole code in a single JS file and that acts as the entrypoint / script tag 


The compilation to a single file, is done by using webpack or similar tools that converts whole of your react code to a single JS file !

```html
In the script tag , you just have to mention <script id='dist/main.js'/> !! That is the compiled main.js file 
``` 

## Bundler

All the files get compiled to JS , even css all gets compiled to JS and then created in a single file in the dist folder , and in HTML you just have to refer to a single div in which you add the <script src="bundle.js">   

That's it ... This is how you use react in a webview in a vsc extension 

In Webpack, a loader is a tool or plugin that processes and transforms files (like JavaScript, CSS, images, etc.) before they are bundled into the final output.

### Loader 

A loader in Webpack is a mechanism that allows you to preprocess files as they are imported or loaded into your project. Loaders transform these files into modules that can be included in your application's dependency graph. For example:

1. Compiling TypeScript to JavaScript.
2. Transforming SASS/SCSS to CSS.

### Types of loader 

1. babel-loader
2. style-loader
3. css-loader

What they do is convert the complex coding file code to simple 2 (css , js) / simplify them

complex file ( like typescript , css )  => loader => (easy files) => webpack ( makes it to a single file) 

Webpack makes it all in a single file .. 

### Custom Scripts
Using `npm run` we can run the custom scripts which we define inside the scripts section in the `package.json`   

## Extension.js
Tree views != Webview , they are both seperate things 

[Tree View](https://code.visualstudio.com/api/extension-guides/tree-view) 

[Web view](https://stackoverflow.com/questions/77978393/visual-studio-code-webview-provider)


## FLOW FOR AN EXTENSION 

UI has the buttons, textarea , dropdowns , etc .. 
The user interact with these values and these values are linked to the 'postMessage' .. the interactions with the frontend are sent to the backend i.e. extension using the post Message ( kind of a post request )  

`package.json` : the commands name for the extension (register command / webview). In package.json we register the commands name that the project has with the description of what it does  ( like the endpoint declaration )   

`extension.js` : In this we register the command mentioned in package.json and tell the functionality of that command, 

webview.onDidReceiveMessage() , this takes in the data and classifies using switch-case to tell what to do .. 


## Typescript 
Interface in React = enum in python 
Same as JS , but with typecontrol ( it helps to define type ) 

## Sandbox env
Name originates from children playing in a safe env is called sandbox. 

We create a sandbox environment, that means , we allow access to security boundaries , limited API and controlled interaction 
We block OS access , Hardware access , Network calls etc .. 

So one workaround is: 
Call an API, that interacts with OS and returns to you the required stuff ( like you can't take a screenshot from controlled env so expose an API that takes the screenshot and we return the output from that )

But the sandbox env's differ that means: 

Browser tab : Chrome ,Firefox ... no FS/OS access
Plugin System : Adobe , OnlyOffice, VScode ... Host-app API only
Language VM : JVM , .NET 
Container : Docker ... OS-level
Full Virtualisation : VMware , KVM 


Web based sandbox : using param in iframe named `sandbox`

We create namespaces to avoid bumping to the same function-name, so it's used to create seperation 

`unshare` : linux command to create a new namespace , they are process level 
`docker run` : also uses unshare internally


```
unshare --pid --uts --ipc --net --mount --fork /bin/bash


# Confirm isolated hostname
hostname  # Returns: sandbox

# Check available interfaces
ip link show  # Only shows lo (and it's DOWN)

# Verify /tmp is separate
df -h /tmp  # Shows tmpfs mount

# Check PID namespace
echo $$  # Should return 1 (init process for namespace)
```
































