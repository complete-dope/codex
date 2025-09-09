---
draft: false
date: 2025-04-25
title: Dual booting arch setup with windows
---

So now we have arch installed and need to dual boot with windows ( obviously for nvidia broadcast to work and cracked premiere pro to work there, wine just doesnt work for me )

**Partition** 
The first role is to make a partition , a virtual partition in your drive so that we can install windows there .. 
How to make a partition ? 
> if you want to part a drive that is mounted ( means that you are using it or is accessible to you ) then it cant be done as you need to unmount it first to create a partition ..

Create an arch bootable , change the boot order then it auto prompts to start with the arch installer , and once you are in there (basically now your ssd is not mounted its just placed there and arch can detect it as it detects other hardwares) , you can also list all the drives using 
`lsblk -f`

This lists all the drives / partition detected in the system

Then create a partition, 
1. Check using `e2fsck`
2. resize using `resize2fs`
3. sudo parted `disk-name`
4. `mkpart` to create a new partition

Workaround if the partition is not created succesfully 
`sudo pacman -S testdisk` : handy tool great at recovery / failure task 


**BOOT problems:** 
Intel RST ( rapid storage technology)  
Need to sometime disable this VMD ( virtual management device )


**NTFS using gparted**
Once partition is done then we need to make it to NTFS file system 
>  use gparted  


Once done start installing windows, and then add missing drivers if any and update the system 

**Reinstall back GRUB**

Grub : grand unified bootloader 

you are done with windows, now insert back arch iso and bring up the grub 

> arch bootable
> minimal installation till root@archiso
> then mount the relevant partitions at /mnt and /mnt/boot   
> then chroot into the system ( change root )
> reinstall grub there
> generate a config file and make sure it says something like 'windows boot loader' , uncomment relevant lines


Its done !!
Now you have arch with windows   




















