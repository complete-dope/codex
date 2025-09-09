---
layout : post
date : 2025-09-10
title : Learnings from astro app AS 
---

## Load balancer 
Building up load balancer 
Get the input metrics sorted : 
1. Latency
2. Cost
3. Uptime

Fit a linear equation that matches your score function : 
`SF = W_1 * latency + W_2 * cost + W_3 * uptime `
where W_1 + W_2 + W_3 = 1


## AB testing 
Create 2 buckets 80/20, whenever a user comes check if its in A bucket or B bucket or assign it one based on the rule so now it becomes a cohort to choose from and see which is a primary and which is a secondary users and experiment with them accordingly !! 

Store that in redis for faster retrieval, 


## Systems thinking 
See in future what this particular small change will go out to the make a big difference   
Its more of a combo of chaos theory and seeing historic patterns 


## Redis storage
Redis is an in-memory storage compared to others DB's like : Cassandra , etc which are not ..  

Internally redis uses a hash function , so 

```
key -> hash function -> bucket_number

inside bucket_number we store as a linked_list
```

Load factor ~ 1, if the load factor increase it increases the hash table size = bucket size ( its in power of 2 always) , so redis tries to keep teh no. of items added / total no. of buckets = 1 , that way we can reduce the time complexity   


The hash function used is the murmur/Sip hash 

### How redis works ? 
redis stores the value in key-value pairs as shown above , and the storing method is same, if collision is found ( means same bucket for multiple items ) the newer items are stored in a linked-list manner 

To find the index / bucket index : `index = hash & (sizemask)` where `sizemask = table_size - 1`

And if your value is a list, ( `'bucket_AB_test_80_split' : List[Profile_ids]`) then its a O(n) time for matching it , rather use sets ( O(1) uses same method as explained above , or use sorted sets ( O(logn) )

So list takes on average around O(n) and with set you can reduce that to O(1) , that is an optimization to look to  



## Disk based storage systems 

Like Cassandra, it doesnt do IO based operation to improve efficiency / lookup / search 

Replication factor means how many times the data will be replicated 

For a 6 node cluster with a replication factor of 3
```
Node1: owns A, D, G  | replicas of B, E, H
Node2: owns B, E, H  | replicas of C, F, I
Node3: owns C, F, I  | replicas of A, D, G
Node4: owns J, M, P  | replicas of A, E, I
Node5: owns K, N, Q  | replicas of B, F, J
Node6: owns L, O, R  | replicas of C, G, K
```

So more the RF more the fault tolerance

Uses LSM Tree Structure to store data 


Create schema for Data, data comes in find relevant nodes ( depends on RF value ) , then add that to commit log of each of the node and also add to mem table , once they both ack we say write is done, then once memtable is filled up / fully filled flush that to SSTable , ( memtable stores in sorted order) 

when flushed to SStable , we also keep : 
```
Bloomfilter.db 
summary.db
filter.db 
```


so at search time we check these above 3 .db first and then search the disk for the actual file 
























