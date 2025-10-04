---
draft : false
tags: ['react-native', 'mobile-app'] 
title : learning React Native
date : 2025-07-13
---

# Building a React Native App 

Very similar to react.js app , simple libraries , nothing much    

## CONCEPTS : 

### ScrollView vs Flatlist
Flatlist is useful when you want to just show a single screen on the scroll ( like reels ) it loads the item present in the list and show them as per scroll, just update the list and your can continue scrolling indefinitely , ScrollView is useful when you dont want to bifurcate between scrolls, and you want to load the whole data, for continous app screen pages like (home screen ) scrollview is useful, else for reels Flatlist is useful    

### Fonts
Take fonts from google app store , that is download the relevant font and then use it 

### Storing data in Cache

It's a simple key-value pair used to store value that can be added / updated in the cache !! 
[Async Storage](https://www.npmjs.com/package/@react-native-async-storage/async-storage)


## React native picker 
Picker is not smooth to state changes , so better is to first add an isloading, boolean and see if its loading , if its loaded then take the updated value , this is the only solution I can find for using the native picker !!   


# How to not release a build everytime a feature is released

## OTA updates
Over the Air updates , the components , non native things can be shared deployed using checked / verified OTA

## Server side UI 
Server Side UI, that is the whole UI of the app is based on a json file and these UI components are added in it .. 

So a .json file that is rendered from an API that is fetched to get the UI of the app and crud operations on UI are done using the OTA updates !!  












