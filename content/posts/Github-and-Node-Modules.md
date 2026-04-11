---
draft: false
date: 2025-03-26
title : Node modules and Github 
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

### Checkout to a branch that is available on remote and not on local   
```
git checkout -b branch_name
git branch --set-upstream-to=origin/branch_name
```

better method for same, 

```
git checkout -b branch_name origin/branch_name
```

### Check the status remote of this branch 
`git status -sb`

### Pull ⚠️ 
Pulls the changes from remote branches to your local branches (use carefully can be destructive) 

`git pull origin` : Pulls changes from same remote branch. eg: If on branch `feat1` this command pulls from `origin/feat1` 

On the other side we can also do 

`git pull origin/main`: Pulls changes from `origin/main`. eg: If on branch `feat1` this command pulls from `origin/main` and this can lead to conflicts as well 

So we define a pull strategy and we have 2 options to choose from : 

`git config pull.rebase false`: makes the default git behaviour as pull and merge (your commit history is not maintained in this)
`git config pull.rebase true `: pulls the latest commits to the branch and add your commits on top of that 

Before :
```
         A---B---C   (origin/main)
        /
   M---N---X---Y     (your local main)
```

After rebase : 
```
   M---N---A---B---C---X---Y   (your local main )
```

this takes changes from origin first and then on top add our changes over it 

Rebase false
```
         A---B---C
        /         \
   M---N---X---Y---M'   ← your branch now
             ↑
        merge commit

```

### Dropping specific files changes 
```
git checkout my-branch // this is the original branch in which we need to make the changes
git checkout origin/main -- path/to/unwanted/files // take these files from the origin/main branch 
git commit -m "Drop unrelated changes" // commit to the original branch

// then raise a PR for it but remember in dev you need to revert back to the prod one for this 
```

### Problem when working in a large team where many are working on same features 

Problems: 
1. many people working on same file, potential conflict
2. you working on a seperate feature , others working on other features
3. you deployed branch on prod and then rebased the branch, that leads to fast-forward errors cause remote feat branch doesnt know about these rebased commits   

### Rebase : 
Lifts your local commits off the branch and puts them in a temporary "holding area."  
Resets your branch to match the latest commit on the branch from where you are rebasing e.g. : `(feature) git rebase origin/main` , here first we will have changes from origin/main and then we get changes from feature branch on top of this. 

Replays your commits one by one on top of that new base branch

Industry standard is to first do rebase, fix errors that come in between and jsut stage those dont commit , and continue your replay using this command : `git rebase --continue` that and then do squash and merge, else if no conflict comes then change this part  

`Rebase , Squash and Merge and then Rebase `: deadly , first you rebased (nice) , then did squash and merge ( nice , and now a new commit was created that your feature branch has no idea of and then you try to do rebase ( git tries to rerun your commits over top of the new commit and results in massive conflicts ) so you start getting merge conflicts in that part. 

Once a branch is squashed , retire that branch dont use that same branch again

Sometimes rebase gives issues that is it gets a merge conflict then you need to continue it : 
`git rebase --continue`

and if you want that these changes / resolution get resolved 
`git config --global rerere.enabled true` or `git config rerere.enabled true`


### Merge:   
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

To solve problem 3, we need to push but with lease so use : `git push --force-with-lease` , this makes sure to push changes only if no one else has pushed in your branch in that time 

### Stashing

Using stash you save your current changes and then go back to some other branch and pop those changes there 

stash push : `git stash push -m '<message>' ` , pushes into stash
stash pop : `git stash pop`  , removes it from stash
stash apply : `git stash apply stash@{0/1/2/..}` , does not remove the stash
stash list : `git stash list` 
stash drop : `git stash drop stash@{0/1/2/..}` , deletes a stash manually 
stash clear : `git stash clear` , deletes everything 
stash untracked files as well : `git stash -u`
stash parts of a file : `git stash -p ` , not useful 


### Reseting changes
Very helpful feature, lets say you commited some files and made 10 commits on top of main and now you need to merge those changes on main and there you realized that some files are not in correct shape like you wanted so there this reset comes in handy 

`git reset --soft <commit-id>` : this takes all file changes that you had above this commit-id in the staged changes place and then we can compare and see that with the other branch and get to know what all changes you did in all those commits 

never do `git reset --hard <commit-id>` 

### Orphan branch 
So if you need to create a branch in a repo and you dont need historical commits and baggage from it, the best way to start this a clean repo with all changes in staged is to do `git checkout --orphan <branch-name>`



