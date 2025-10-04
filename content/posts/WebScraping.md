---
layout : post 
date : 2025-02-10
title : Web Scraping
---

# Learning to scrape web 

The modern web is dynamic, so we need to use a tool that can atleast scrape dynamic content and get the relevant results

The static scraping sites like : Requests , cheerio, they dont work for dynamic site 

Dynamic sites scraping : Use tools like playwright ( inject javascript on site load and get the data ) 

But modern sites catch this data (playwright) 

## We have something called Chrome Devtools Protocol ( CDP )

CDP is a low-level, JSON-based protocol that allows external tools to interact directly with Chromium-based browsers (like Chrome and the Chromium engine used by Edge).

When you launch a Chromium browser with remote debugging enabled, it opens a WebSocket server. Automation tools (like Playwright or Puppeteer) establish a WebSocket connection to this server.
try this with `firefox --remote-desktop-session=9222`

CDP is a connection between your automation tool and browser 

Chrome dev tools also uses this same protocol to debug .. 

Capabilities:


Real-time browser instrumentation and control
Access to network traffic, DOM, JavaScript execution
Performance profiling and debugging
Access to Chrome's JavaScript console
Ability to capture screenshots and generate PDFs


Use cases:


Browser automation
Testing and debugging
Performance monitoring
Network interception
Security testing



### Web driver protocol 

WebDriver is a standardized protocol for browser automation, part of the W3C WebDriver specification. It's language-agnostic and works with multiple browsers. Key aspects include:



Difference between the 2 : 

Think of browsers like a car. Now, imagine you want to control this car in different ways:

Webdriver: 
* This is like a standard remote control that works with ANY car (browser)
* It's simple: "go forward", "turn left", "stop"
* Works with Chrome, Firefox, Safari, Edge - pretty much everything
* It's slower but very reliable
* Perfect for basic stuff like:


Useful for : 
Clicking buttons
Filling forms
Taking screenshots
Basic navigation

### Chrome Dev Tools: 
This is like having direct access to the car's engine and computer system
Only works with Chrome/Chromium browsers
Much faster and more powerful
Can do advanced stuff like:

Monitor network traffic
Debug JavaScript
Modify page content on the fly
Check memory usage
Intercept and modify network requests



### When to use what ? 

Use WebDriver when:

You need cross-browser support
You're doing basic automation
You want stable, simple code


Use CDP when:

You only care about Chrome
You need advanced features
You want faster execution
You need to debug or monitor deeply

CDP is a low level protocol, and it uses webdriver which is a high level prop to actually control the browser.

## So at end we are left with 2 options .. 

In terms to getting detected as automated
Webdriver most easily detected > then comes playwright > then comes request 

1. If the site is not anti bot detected : make a request using our normal playwright .. 
2. If its antibot detected : make a request using requests library .. that will pull static data .. 

Simple lets live with it .. 
























