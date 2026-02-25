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


`git config pull.rebase false`: makes the default git behaviour as pull and merge (your commit history is not maintained in this)
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

in one branch , we have config.json(dev) files and in another branch we have another config.json(prod), so while merging this raises merge conflict that can be easily sorted , but in PR this gets reflected so a better way to do this is for dev , use dev config , then when merging from prod use prod config, 

```
git checkout my-branch // this is the original branch in which we need to make the changes
git checkout origin/main -- path/to/unwanted/files // take these files from the origin/main branch 
git commit -m "Drop unrelated changes" // commit to the original branch

// then raise a PR for it but remember in dev you need to revert back to the prod one for this 
```

# Problem when working in a large team where many are working on same features 

Problems: 
1. many people working on same file, potential conflict
2. you working on a seperate feature , others working on other features 

## Best way to merge branches together
`Rebase` : 
Lifts your local commits off the branch and puts them in a temporary "holding area."  
Resets your branch to match the latest commit on the origin (origin/main).  
Replays your commits one by one on top of that new base.  

Industry standard is to first do rebase, fix errors that come in between and jsut stage those dont commit , and continue your replay using this command : `git rebase --continue` that and then do squash and merge, else if no conflict comes then change this part  

`Rebase , Squash and Merge and then Rebase `: deadly , first you rebased (nice) , then did squash and merge ( nice , and now a new commit was created that your feature branch has no idea of and then you try to do rebase ( git tries to rerun your commits over top of the new commit and results in massive conflicts ) so you start getting merge conflicts in that part. 

Once a branch is squashed , retire that branch dont use that same branch again


`Merge` : 
Keeps the commits seperately so that everything can be known like when was this started , its first commit and helps in keeping all the history
`--no-commit` : stages all the commit but doesnt create that final merge commit.    
`simple merge` : merges the branch and adds that final merge commit as well to it.    

BEST METHOD : 
```
Many modern tech companies (think Google, Meta, and high-growth startups) prefer Squash and Merge.

Why they prefer it: When hundreds of developers are pushing code daily, a "Normal Merge" makes the main branch history impossible to read. It becomes a jungle of tiny, meaningless commits.

The Workflow: 1.  Dev works on a feature (50 messy commits).
2.  Dev Rebases locally to stay up to date.
3.  Dev Squash-Merges into main.

The Result: The main branch looks like a clean, professional list of features:
Feature A added -> Feature B added -> Bug C fixed.
```
