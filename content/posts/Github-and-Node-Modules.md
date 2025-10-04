---
draft: false
date: 2025-03-26
title : Editing the node modules and github commands 
---

A noob method of playing with Node modules  

Steps to follow to update code from a node_module like `d3-force` directly !!

So what we get in a node_modules are the build folder that dont support HRM (hot reload module) so we have to make changes and then build that folder using the rollup module ( maybe we can use other libs also !!) 


Steps: 
1. Make that change in the file that you want !!(be it debugging statements or logic change  )
2. Install the required files, using `npm install` and if required change the package manager from yarn to npm    
3. Then lookup to package.json and you might find a function named "prepublishOnly"
4. Run the command `npm run prepublishOnly`, If the command requires the some folder , build those or remove those folders from the npm command  !!  
5. Then once you fix the errors , got the errors solved , then we build this file using `npm run prepublishOnly`
6. Then run the development server using the `npm run dev -- --cache`

This is how to rebuild a node-module  !!  



## Github commands 

```
git checkout -b branch_name
git branch --set-upstream-to=origin/branch_name
```

or, 

```
git checkout -b branch_name origin/branch_name
```

to check the remote of branch we get, 

`git status -sb` : helps to find the remote branch 


`git config pull.rebase false`: makes the default git behaviour as pull and merge 
`git config pull.rebase true `: pulls the latest commits to the branch and add your commits on top of that 

```
         A---B---C   (origin/main)
        /
   M---N---X---Y     (your local main)

```

Rebase false
```
         A---B---C
        /         \
   M---N---X---Y---M'   ← your branch now
             ↑
        merge commit

```

Rebase True
```
   A---B---C---X'---Y'   ← your branch now

```




