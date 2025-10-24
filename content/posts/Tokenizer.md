---
title : Tokenizer
draft : false
tags : ['tokenizer', 'byte-pair encoding', 'encoding']
date : 2025-10-24
---

# Encoding
Its a way to send / transmit information    
The transfering of data across internet is done in byte stream and we have a defined format to encode those in a byte stream 
Like if its a text data then we have `utf-8` encoding , if its a audio file its `mp3` encoding, image file its `png` encoded like this we define encoding for all dataformats and on client side decoding happens to output it in a compatible way to the end user   


# Tokenizers 
So there have been research papers that are dealing to directly feed in the byte streams to the model and avoid tokenization at all : https://arxiv.org/pdf/2305.07185  

So one naive way to tokenize it to seperate out all the characters and then do an bottom up to get the encoded pairs and other one is to seperate out using regex and then treat each word individually and use it to train a model !



