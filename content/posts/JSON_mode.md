--- 
title: Adding JSON mode to any model and that too without prompts  
date: 2025-10-10
draft: false
tags: ["json-mode" ,"bnf" , "cfg" , "context-free-grammar"]
summary: "Learning about how to add JSON mode to any model and dont just solely on prompts"
--- 

## Grammar 
So as in english grammar tell whether the sentence is correct or not ..  

`I am eating` = <subject> <conj> <verb>
`eating am I` = this is incorrect as the grammar is not met 

Similarly all coding languages have a grammar that they use to check syntax errors / parsing .. 
like python : https://docs.python.org/3/reference/grammar.html 


## BNF 
Its a form of writting that grammar
```
<expr> ::= <term> | <expr> "+" <term>
```

`:=` means 'is defined as'  

so this is how lexer in compiler breaks it up in tokens and then verifies its order



## CFG ( context free grammar )
This means the value / expression on the left side does not depend on anything : 

```
Expr → Expr + Term | Term
Term → "1" | "2"
```

Top line is read as expression equals to (expression + term) or (term) alone 
Here I am saying expr is independent of whatever comes on the right side 

so this can become '1+2+1' 

Example of context-dependent grammar would be `aA → ab` this `A` can only be replaced by `b` only 

## Json / response-format mode

In LLM's we dont force on next token generation so that they can freely write it out ... but to get a particular json output from a model we can enforce this CFG to it using : https://github.com/ggerganov/llama.cpp/tree/master/grammars

So we this format called as GBNF ( is a grammar format only that's required by ggml models ) and for we can create an GBNF for any pydantic class that we want using https://grammar.intrinsiclabs.ai/ , just convert your python schema to typescript once  

More details in this page : https://til.simonwillison.net/llms/llama-cpp-python-grammars

## Learning to write GBNF (as no LLM / online editor is helping) 

So everything start with `root::=` that is defines the root architecture and in this we can define all variables that will help in this 
and every expression begins with a inverted-commas  `"<single-expression>"` , and each expression is in itself a wrapped in inverted commas 
### 1st example


```
root ::= "{ \"status\" : " string "}"

so these are 3 expressions break-down as

term-1 : "{ \"status\" : " --> this tells model to begin with this {"status" : 

term-2 : string --> string as defined

term 3 : "}" --> to end with } 
```

String is defined as: 

So lets do this step wise: 

First I am defining the string variable name and initializing it empty with inverted commas
```
string ::= ""
```

Second, add internal value to it 
```
string ::= "\"active\" | \"inactive\""
```

this means it can be only a literal value
Its comparison in pydantic class is

```python
class JSONSchema(BaseModel):
  status : Literal['active', 'passive']
```


### 2nd Example
rather than literal it can be any string
```
string ::= "\"" ([ˆ"]*) "\""
```

so this is also made up of 3 things
part-1 : "\"" --> should start with single inverted comma " 
part-2 : ([ˆ"]*) --> here we can write any string value
part-3 : "\"" --> should end with single inverted comma "


