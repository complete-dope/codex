---
title : High level design 
tags : ['high level', 'event driven', 'system engineering', 'design patterns']
---


## Event driven architecture
This works in a queue based system, that is, we have a producer that is doing some operations and sending that to broker (kafka / mq) and then on other side of message queue are consuming it. 

These queue are usually designed for kb-mb of message sizes only 

and assume we get 1000's producers producing and sending messages to brokers, kafka handles this internally using `epolls` that sleep when nothing is producing and as soon as we get somethign on network stack it does an NIC interrupt and wakes up a sleeping thread and that is passed on to the CPU for processing 

