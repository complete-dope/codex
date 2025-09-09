---
title : Learning NMAP
date : 2024-02-29
layout : post
---

# Learning Nmap 


Basics: 

* Ping scan : Pings all devices on a subnet network => `nmap -sp 192.168.1.0/24`, which sends all devices ICMP echo request (Internet Control Message Protocol Echo, its a request to check 'Are you there') , tools like `ping` sends these requests


* Single host scan for Ports : examine a single host for 1000 most common ports .. ( 80 : https , 22 : ssh ) .... `nmap 192.168.1.100`  // `nmap scanme.nmap.org`


* Stealth Scan : It doenst complete the handshake, it sends a SYN , the hosts sends back an SYN-ACK , but that ACK is never registered back .. closes the port sends back a RST so this connection is never actually triggered. It reveals all open ports in a machine ..
`nmap -sS 192.168.1.100`


* version are important, as it tells which version is installed and we can exploit the vulnerabilities of that version ...
`nmap -sV scanme.nmap.org` / `nmap -sV 192.168.1.100` : scans the network for versions .. ( which version ?? version of services running on open port

Once you get the version being used, then exploit the vulnerabilities using tools like metasploit 


* OS scanning : Its tells the OS being used on the machine, not always correct 
`nmap -O 192.168.1.1`


* Aggressive Scanning : tells OS detection , version detection on open ports , script scanning , traceroute


* Scanning multiple host : like in a for loop
`nmap 192.169.1.*` or `192.168.0.1,2,3,4` 


* Port scanning : For single port `nmap -p <port-number> 192.168.1.10` // `nmap --top-ports 10 scanme.nmap.org`


* verbose mode : `nmap -v 192.168.1.1` tells what happening behind the scenes  


* NSE (nmap scripting engine) : helps to run the scripts , you can either use pre existing scripts or can create your own ones using Lua ..   



OUTPUTS 


-sS 

'''txt 
Nmap scan report for 192.168.1.7
Host is up (0.017s latency).
Not shown: 991 closed tcp ports (reset)
PORT     STATE SERVICE
53/tcp   open  domain
80/tcp   open  http
139/tcp  open  netbios-ssn
443/tcp  open  https
445/tcp  open  microsoft-ds
3261/tcp open  winshadow
5000/tcp open  upnp
5001/tcp open  commplex-link
5357/tcp open  wsdapi
'''












