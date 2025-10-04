---
draft: false
date: 2025-02-28
title: Learning about Hardware and ROS and controllers
---

# Learning Hardware stuff 

Nice_Chat : https://chatgpt.com/share/67c0be3f-ca80-8009-b271-f3c1f626edbf


## Learning microcontrollers

Esp32 : One of the most popular microcontroller that is inbuilt with bluetooth and wifi card ! 
ATmega328P : Microcontroller that is used in Arduino UNO 

GPIO  : general purpose input output ( can act as input or output pin, depending on the code and its usage ) 

`What is a firmware ? `
Firmware is a low level control over hardware , directly written in the hardware RAM, to the memory ( embedded in metal ) kinda like a kernel that manages / controls the microcontroller  

Microcontroller is the main part of a board / circuit that has embedded cpu , memoy etc and it has pins exposed ( lets say 15 pins ) and each pin can be classified as an IO pin or as an data pin. Now we made microcontroller so small but the IO devices and other modules also need to be mounted on the PCB as the wires are etched in that PCB so lets say for wifi module to work with microcontroller we need to expose a wifi pin and then the wire etched in pcb need to be connected to the wifi module and then it can be used same goes for IO device and we have that whole board .. aka development board , microcontroller board 

The development board has pins exposed for external connections , power supply etc and the real struggle comes in making electrical connections / electronics is correct and make sure you dont blow up things by giving them extra voltage etc .. 

So the electronics parts is a bit tough part !!

[Understanding arduino pins](https://www.youtube.com/watch?v=bniUECtJkeU)

GND : ground pin it allows current to flow out of internal circuit into the Ground 

### Learning ESP32 ( connecting things and playing around )
gpio 2 is connected to inbuilt LED so if you use gpio 2 as the input high source that would mean the inbuilt led would also glow whether you want it or not .. 
so use some other free gpio like gpio ( 4, 5, 13, 14, 15) etc and get your work done 

Buttons ( omron b3f ) , 4 legged tactile switch when pressed creates and internal connection, pressed state is called LOW state , and unpressed state is called HIGH state. 

### Serial Port



#### Power supply Ping 
Using Power pins that are attached to arduino with voltages 

#### Analog Pins
Using Analog pings


#### Digital Pins
0 to 13 pins, 
Can have only 2 modes , 1 or 0 , 1 is high that means 

Digital pin can either be in input mode or in output mode and you need to configure that mode in code !!

Input mode is used to read data from sensor 
Output mode is used to write data to actuator 

Then high state corresponds to 5 volt for high and 0 volt for low 

PWM ( pulse width modulation ) for fractional voltages by changing the width and those are with '~' next to there number .. These pins can be used normally in binary mode or in pwm mode only in output mode 

```
digitalWrite( PIN , HIGH ) 
digitalWrite( PIN , LOW ) 
```

#### Interrupt Pins 



#### Analog Pins
useful to read values that cant be just 0 or 1
Assume you have a potentiometer, with a digital pin you could just know the potentiometer is at max or min and nothing else .. 

It reads the analog value from a sensor, lets say it reads 2.5 volts, so your adc ( analog digital converter) changes it from 0 to 1023 
0 being 0 volts
1023 being 5 volts 
614 = 3 volts 

Analog pins can only be used to read analog values it cant write it 

3 Communication protocols between pins: 
uart , spi, i2c 

uart : connected to computer using library 



[esp32 vs arduino](https://www.youtube.com/watch?v=RiYnucfy_rs)

### Antenna ( old and new antennna's ) 



### PCB Antenna 


## Powering the microcontroller !!

`How to power a microcontroller ? What are the appropriate voltages to supply ? and what does those pins means ? `

Ground pin : For electricity to flow we need a closed circuit, if there is no ground connection electricity has nowhere to go and circuit wont work 

all components in a dev board need a common refernce point to measure voltages 

current goes from positive(+) to negative(-) and electrons flow form negative(-) to positive (+)

Battery positive is like a hilltop with extra energy and negative side is like a bottom of a hill and electrons roll down the hill

Think of an electronic circuit like a water system:

    Battery/Power Source = Water Pump (pushing water up)
    Wires = Pipes (carrying water)
    Resistors/LEDs/Motors = Water Mills (using energy)
    Ground (GND) = Riverbed (where water flows back)

If you don’t have a path for water to return (ground), the water won’t move, just like electricity won’t flow without a ground.



For getting a reference value we use GND, we can use lets say 3V as the ground but now the potential difference will be battery voltage (10 V) and the devices will be getting only 10-3 = 7 volts !!

More comes to play in series and parallel connection ..  



### Arduino Pins 
If you require periodic controlling of something without manual interverntion then we require microcontrollers in that !! and they these controllers just like human , acts on behalf of them 

Digital Pins : The output from these are 0 and 1. We require to set pinMode to either OUTPUT and INPUT output pin can output 2 things 0 or 1 . 0 (zero) means low  , 1 means high voltage !

GPIO registers ( general purpose input output ) registers and DDR ( Data Direction Register )

Internally arduino uses MOSFETS working in combination to achieve this state ! 

so first we setup pins / init them and then we use loops to run in those or different configurations  


# Hardware and drivers and how do they talk to each other !!

Just imagine if the kernel maintainer has to keep up with every new development / app coming in the market, then it will not be possible 
So what they did is, currently we have around 100's of system call ( that are essential for system boot-up and using the system ) and an exposed sys-call called `ioctl` , `sysctl` that extends the kernel to run something called as 'device-driver'


The device-driver is the code that interacts with the hardware from the kernel space and is maintained by the company providing it .. but the code provided is not easy to understand for the developers to be able to work with it .. so they provide a compatible user space library that normal developers can use and interact with the underlying hardware ... 

And there role is to maintain the device driver + the user space and bring regular updates to it ..  

Application → [libdrm] → [ioctl → i915 kernel driver] → Intel GPU Hardware


















