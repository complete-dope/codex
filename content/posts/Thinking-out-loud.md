---
layout : post
date : 2025-05-11
title : Thinking and fiddling out with random ideas
---

# Learning new ideas and fiddling with them 


## When you visit a site what happens ? 
*The html , js , css gets transferred to the browser using http request and then the chrome engine runs that and display the content 


## How does image upscaling works?


## How does real time stt works ? 
We store the 500ms chunk in /tmp file , send it to transcribe and then repeat this until we close it .. or a VAD (voice activity detection) is used to check till when the voice is detected ..  


## Network differences between a sse and stdio protocol ? 

SSE : Its a long lived HTTP GET request , that the client sends to the server and server sends the data back to it ..

STDIO : Its a local IPC ( inter process communication ) , Standard Input/Output , its a way for programs to read and write data , typically in command line or process-based environment , 

stdin : input stream ( keyboard , input mic )
stdout : output stream  ( printed output , transcription result )
stderr : error messages 

Its for internal communication , lets suppose a server (whisper model) -> audio chunk -> subprocess transcription -> transcribed output -> send to frontend using SSE or websocket


Now lets see an LLM request: 

1. /post request from frontend
2. sends that to LLM 
3. receive chunks -> send back to frontend via SSE

Hosted LLM sending chunks : 
Recieves prompt -> predicts tokens clubs and sends it back -> (sse) here the client is the backend that called that LLM ..  

Local LLM sending chunks :
Recieves prompt (stdin) -> predicts tokens -> sends chunks (stdout) -> stdio operation 

```python

# Start local LLM binary
llm_process = subprocess.Popen(
    ['./llama.cpp', '--model', 'mistral.gguf'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Send prompt
llm_process.stdin.write("Tell me a joke\n")
llm_process.stdin.flush()

# Read output line by line (like tokens)
for line in llm_process.stdout:
    yield f"data: {line}\n\n"  # <-- SSE back to frontend


```
















