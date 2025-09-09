---
layout : post
date : 2025-07-18
title  : Learning Go-lang
---

## Learning GO to get the minimum latency and get things right from start .. 
[Go docs](https://go.dev/doc/tutorial/getting-started)  
[Go Tour](https://go.dev/tour/list)

The whole file needs to become a package and to make it whole a package we wrap that up in `package main` and this makes the whole code as a single package that is then converted to binary 

We create package to help us import those in other files and the importing helps in code seperation, else we would have to write the whole code in a single main.go file.. 


### Imports 
The internal imports are done using: 

```
import (
  "fmt"
  "net/http"
  "github.com/gorilla/mux"
)
```

so these are the imports as we made in the python and to get these imported sorted we have to take 2 steps

```

1. `go mod init <package-name>` : can be named anything, we need this to make sure the module is go.mod and public and anyone can download and use the functions / classes listed in it ..

2. `go mod tidy` : this is like pip install -r r.txt where in python we write the library names in r.txt here we write the libraries in code itself and we they gets installed and added as require in the mod file
```

`func main()` is the function that gets called by default as soon as we compile the app 

`go run .` : runs the file 


### Multiple modules 

If each module in a separate directory, you will need to basically convert that package from external link to the local one :

```
~/go-dev/
├── greetings/
│   └── go.mod (module example.com/greetings)
|   └── greetings.go 
└── app/
    └── go.mod (module example.com/app)
    └── app.go 
```

So to use greetings.go in app.go we need to import in this way in the 
`go mod edit --replace example.com/greeting=../greetings`

also the package name should be same as the directory name .. 

### Variable declaration 

```
var name string
name = "Mohit"
--
name := "Hello" //Here the type declaration is taken from the rght side of the string 
```

`list` declaration:

```
message := [] string { 
  "Hi 1", 
  "Hi 2",
  "Hi 3",
}
```

### Error Handling 

send the value , error with it  .. 

so the return is a tuple ( message , error-code ) , error return nil means no error , else send an erorr using : 

```
import "errors"

if name == "":
  return "", errors.New("empty name")
```

## Range keyword
used to iterate over 
```
Array, 
Slices, 
Strings, 
Maps,
Channels
```

## Make
`map_init := make(map [string]int )`
`slice_init := make([]int, 5)`
`chan_init := make(chan int)`


## GoRoutines 
goroutines are like threads, assume that they are threads and you can goroutines will work, 
new goroutines are created using the 'go' keyword (prefix) , 
any function can become a goroutine just add the go keyword in front and doesnt require function coloring 


## Channels
channels are way of communicating before 2 goroutines, data transfer between goroutines are done using these channels, 

Each channel is a has a particular type of data-structure / data-type that is called `Element` type 

```
// so its a integer data channel means integers can only passed through this channels 
ch := make(chan int) 
```

`make` : this keyword is used to 3 datatypes (map , channel , struct)

```
ch <- x   ...//a send statement 
x = <-ch  ...//a receive statement is an assignment 
<-ch ...// a receive statement; result is discarded
```

Channels support `close` keyword , which sets a flag indicating that no more values will ever be sent on this channel .. 

Example : 
```
// channels , pipeline

package main

import ( 
	"fmt"
)

// go doesnt allow named functions inside the main() function .. only anon functions can be declared  
func main(){
	naturals := make(chan int)
	squares := make(chan int)

	// Naturals 
	go func(){
		for x:= 0; x<100 ; x++ {
			naturals <- x
		}
		close(naturals)
	}()

	// Squares
	go func(){
		for	x := range naturals{
			squares <- x*x 
		}
		close(squares)
	}()

	// Printer
	for x:= range squares {
		fmt.Println(x)
	}
}
```

### Unbuffered Channel 
by default, the initialised channel is an unbuffered one , that is there is not buffer stored in the channel , once go routines sends a value , it will wait till receiver receives that value

```
ch = make(chan int)     //unbuffered channel
ch = make(chan int, 0)  //unbuffered channel
ch = make(chan int, 3)  //buffered channel with capacity 3 
```

Called as Synchronous channel 

### Buffered Channel 

It has a queue of elements, send inserts at the end , receiver takes out the value from the start, and if the queue is filled the sender no longer pushes the values in .. so that is one drawback

If in an unbuffered channel , goroutine is trying to enter value then it wont be able to add values to the buffer this situation is called `goroutine leak` 









