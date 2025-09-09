---
layout : post 
date : 2025-03-29
title : Learning Celery 
---

## Background task 
Maintaining background tasks specially in python django is damn tough, so we to think spawning some daemon threads and getting that solved is very tough .. so rather we should use celery that starts a complete new process and doesnt maintain any-thing inside main process  

## Celery 

What is the use of celery ? and its comparisons 

asyncio.create_task() : This creates and executes task in event loop ( efficient management of event loop ) 

celery : multiprocessing, creates multiple processes and tasks to run on CPU , that means 

Task Queue : Input is a task , Worker processes constantly monitor for new tasks in the task queue ... (how do they monitor that task )  

Broker : is the intermediary between clients and workers, where clients push there task and from where workers are allocated tasks , eg ; redis broker or rabitmq broker  


### Producer.py

```python 
# producer.py

import redis
import json
import uuid

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def send_task(task_name, *args):
    task = {
        "id": str(uuid.uuid4()),  # Unique task ID
        "name": task_name,
        "args": args
    }
    redis_client.rpush("task_queue", json.dumps(task))  # Push task to Redis
    print(f"Task {task['id']} sent: {task}")

# Example usage
send_task("add", 4, 6)
send_task("multiply", 3, 7)
```

### Worker.py
```python
# worker.py

import redis

redis_client = redis.Redis(host ='localhost' , port = 6379 , db =0)

TASKS = {
    "add": add,
    "multiply": multiply
}

def main():
  while(True):
    json_tasks = redis.blpop('task_queue' , timeout =5) # block when no element to pop
    if json_tasks:
      _, json_task = json_tasks
      task = json.load(json_task)
      func = TASKS.get(task['name'])
      if func:
          result = func(*task["args"])  # Execute function
          redis_client.set(f"result:{task['id']}", json.dumps({"status": "success", "result": result}))
          print(f"[Worker {worker_id}] Task {task['id']} completed. Result: {result}")
      else:
          print(f"[Worker {worker_id}] Unknown task: {task['name']}")

      time.sleep(1)

if __name__ == "__main__":
  main()


```

### Completion fetcher 

```python

output = redis_client.get("result:{task['id']}")

```



