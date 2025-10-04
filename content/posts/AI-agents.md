---
draft: false
title: "AI Agents"
date: 2024-12-23
tags: ["ai", "agents"]
author: "Mohit Dulani"
ShowToc: true
TocOpen: false
summary: "Learning about AI agents and how they are built in real world"
---

![AI agents by andrew ng](https://www.youtube.com/watch?v=KrRD7r7y7NY)


======================

Normal flow ::

LLM writes from first word, start to end with no backspace. and this is not how humans write

======================


With Agentic workflow ::

write essay on topic X
do you need web research
write a first draft
consider what parts need revision or more research
revise your draft

>> for loop to the model with changing prompts

with agentic workflow lower end models tend to perform pretty well and on-par / better with higher end models using no shot

======================
Design patterns for Agentic workflow:
======================

1. Reflection with llm

taking example of coder agent :

```
>> Generate the code with llm , pass it response back to llm and add something like "Check the code correctly for correctness , style , efficiency and give constructive suggestion to improve it."

>> Get the suggestions and update the code

>> ask it to generate some complex unit tests

>> pass those complex unit test from the model and this feedback loop runs iteratively till correct code

>> coder and critique LLM ( with different prompts)
```

======================

2. Tool use
using API calls like we did for the check if API is required or not check for question ...
Like we did for Agastyaa

======================

3. Planning / Reasoning
Hugging gpt model
```
>> You plan a complex request and pick a sequence of request for in order to deliver on a complex task
```


======================

4. Multiagent collaboration

communicative agents for software development 2023 paper

```
>> prompt llm to play different roles at different times so different agents interact at different time
```
specialised LLMs works better

========================

Four AI trends:

1. Agentic workflow consume a lot of tokens and will benefit from faster cheaper token generation ( groq )

2. Today agent are built by taking LLM trained to answer questions and retrofitting then to an iterative workflow. More LLM's will be fine-tuned to use in agentic workflows such as to use tools to plan / reason or to use computers
like claude computer


3. Data engineering's important is rising


4. Text processing revolution has arrived. The image processing revolution is coming, and will enable many new visual AI applications in entertainment, manufacturing, self-driving, security etc




---------------------
![Intro to AI agents](https://www.youtube.com/watch?v=ZZ2QUCePgYw)

Agentic: the degree of autonomous

Agents are agentic to different degree some are more some are less like:
AI completing a task autonomously is agentic and AI researching is also agentic maybe not to the same kind of degree of agentic as first one

Agent that reads email and creates a todo list based on the highest priority task.


Like 2 agents combined, a shipping agent and a customer service agent and check why the package isnt around



------------------------

What's next for AI agents rather than this for loop

Production ready agent


Planning step : Reflection

UX control


------------------------

~[The Future Of AI Agents With Dharmesh Shah | INBOUND 2024](https://www.youtube.com/watch?v=IityUpVVD38)

Amazing talk

AI agent market-place where we have all the agents listed and there to use like : https://agent.ai (linkedin for agent)

One agent can call other agent and work collaboratively

* shows the real future of agents

Building agent and removing all mundane tasks


------------------------

![Vertical AI Agents Could Be 10X Bigger Than SaaS](https://www.youtube.com/watch?v=ASABxNenD_U)


How large that is going to be ?
Saas tools -> last 20 years silicon valley funded only saas category

Obviously good ideas
>> docs , calender, etc where only google, meta , microsoft these won

mass consumer ideas that Nobody predicted
>> uber ,instacart , doordash , coinbase
where startups won

b2b saas companies
>> more billion dollar company , different company for different horizontals


SAME WITH AI / LLM

Obviously good ideas: where top companies will win

mass consumer ideas : like voice assistant etc that companies like google,

B2b saas: big companies will not go to a single horizontal idea as that is a lot of work

LLM system that doesnt obsolute the idea of hiring other professionals for tasks like marketing , hiring, sales etc

** many people have done this .. smart marketing people have timely beaten some of the top specialized people in that field


* Vertical agents means an agent specialised at a single task and performing the best out of it .. (like Agastya)

b2b Saas : 2005-10 consumer applications like emails , chat , maps .. people got used to using these things and that made it easier.


Find yc companies that applied and look for b2b saas companies in there and what they are doing !!



2 AREAS WHICH HAVE BECOME MORE AND MORE EXPENSIVE OVER TIME ARE:

HEALTHCARE AND EDUCATION

WITH AI COMING TO PICTURE THAT CAN BE DONE !!



READS ::
1. https://arxiv.org/pdf/2410.10762v1

2. https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

3. https://github.com/JaynouOliver/list-of-all-yc-s24-startups

4. https://www.youtube.com/watch?v=kHPXbo2OkzA

5. https://www.youtube.com/watch?v=pcWKJ_yvf1M



THE WORLD IS GOING CRAZY OVER AGENTS AND YEAR 2025 IS THE YEAR OF AGENTS

WHAT THE WORLD NEEDS :

SMALL MODELS , EFFICIENT ENOUGH TO DO MOST OF THE TASKS , REQUIRES <10GB MEMORY ,  LONG CONTEXT 1Million FOR ALL
INFORMATION STORED and running locally


Increasing the context size :

1. use semantically compressed text , pass chunks to model , extract info from chunks , concat info then pass it to the model :: noob idea

2.
