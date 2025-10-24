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

So one naive way to tokenize it to seperate out all the characters and then do an bottom up to get the encoded pairs and other one is to seperate out using regex and then treat each word individually and use it to train the BPE model !

So the string  `Hello World` gets broken up as : `Hello` , ` World` and this is splitted into characters as   
```bash
H e l l o   W o r l d 
```

so the `o` from Hello never sees the ` W` from World, that is the use to split using regex for efficient tokenization also this is how humans do it !     



# Dynamically expanding the token-window 
So lets say now open ai wants to add browser use in there usecase and this will require many new token + special tokens so retraining a model from scratch makes no sense rather they just increase the vocab size and keep the base model freezed and only train for these new layers on top of existing ones  

