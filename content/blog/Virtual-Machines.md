---
layout : post 
date : 2025-09-03
title : Virtual Machine  
---

Connecting to a Azure VM using ssh and RDP

Configure a VM based on the specs 

# Connecting with SSH 

Directly connect using ssh create an inbound rule for ssh, you get the ssh key !  

# Connecting with RDP  
RDP , install ubuntu-desktop, 3389 is the port for RDP  , 22 for ssh , 80 for http , 443 for https !

# Azure VM RDP Setup Guide

## 1. Install Desktop Environment (Ubuntu Desktop)
- **Install desktop environment (if not already installed):**
  ```bash
  sudo apt update
  sudo apt install ubuntu-desktop
  ```

## 2. Install and Configure XRDP
- **Install xrdp:**
  ```bash
  sudo apt install xrdp
  ```

- **Start and enable xrdp service:**
  ```bash
  sudo systemctl enable --now xrdp
  ```

## 3. Verify XRDP Service
- **Check if xrdp is listening on port 3389:**
  ```bash
  sudo ss -tulpn | grep 3389
  ```

- **Check xrdp logs for issues:**
  ```bash
  sudo journalctl -u xrdp --no-pager -n 50
  ```

## 4. Ensure Firewall Allows RDP
- **Check firewall status:**
  ```bash
  sudo ufw status verbose
  sudo iptables -L -n
  ```

- **Allow RDP port (3389) in the firewall:**
  ```bash
  sudo ufw allow 3389/tcp
  sudo iptables -A INPUT -p tcp --dport 3389 -j ACCEPT
  ```

## 5. Check Azure NSG (Network Security Group)
- **Ensure there’s an inbound rule allowing TCP port 3389** in the NSG associated with your VM's network interface or subnet.

## 6. Test RDP Connection
- **Use Remmina or another RDP client to connect to the VM:**
  ```bash
  remmina --protocol=RDP --server=<VM_IP_ADDRESS>
  ```

- **Check port availability with Nmap:**
  ```bash
  nmap -p 3389 <VM_IP_ADDRESS>
  ```

## 7. Restart the VM (if needed)
- **Perform a hard restart** if you encounter issues:
  ```bash
  sudo reboot
  ```

By following these steps, you’ll set up and troubleshoot RDP access to your Azure VM with a GUI.
