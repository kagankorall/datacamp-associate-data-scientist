# Intermediate Git

## Course Overview
A follow-up to Introduction to Git focused on branching workflows and collaboration. The course covers creating, switching, comparing and merging branches, resolving merge conflicts, and working with remote repositories through cloning, fetching, pulling and pushing.

## Key Topics Covered

### 1. Working with Branches
- What a branch is and why Git uses branches
- Listing, creating and switching between branches
- Comparing branches with `git diff`
- Renaming and deleting branches
- Merging branches into a destination branch

### 2. Collaborating with Git
- Resolving merge conflicts
- Introduction to remote repositories
- Cloning local and remote repos
- Pulling changes from remotes (`git fetch`, `git pull`)
- Pushing local changes to remotes (`git push`)

## Key Concepts

### Branching basics
```bash
git branch
git switch main
git branch speed-test
git switch -c speed-test
```

### Comparing branches
```bash
git diff main summary-statistics
```

### Merging
```bash
git switch main
git merge ai-assistant
```

### Cloning a remote repo
```bash
git clone URL
git clone /home/george/repo new_repo
```

### Pulling and pushing
```bash
git pull origin main
git push origin main
```

## Course Notes

# Working with Branches

## Introduction to Branches

A **branch** is an individual version of a repo. Git uses branches to systematically track multiple versions of files. In each branch, some files might be the same while others might be different. The default branch is `main`.

```bash
# List all branches
git branch

# Switch to the main branch
git switch main

# Create a new branch called speed-test
git branch speed-test

# Move to the speed-test branch
git switch speed-test

# Create a new branch called speed-test and switch to it in one command
git switch -c speed-test
```

## Modifying and Comparing Branches

### `git diff` recap

```bash
# Show changes between all unstaged files and the latest commit
git diff

# Show changes between an unstaged file and the latest commit
git diff report.md

# Show changes between all staged files and the latest commit
git diff --staged

# Show changes between a staged file and the latest commit
git diff --staged report.md

# Show changes between two commits using hashes
git diff 312gfh42 178fsdf7

# Show changes between two commits using HEAD instead of commit hashes
git diff HEAD~1 HEAD~2
```

### Comparing, renaming and deleting branches

```bash
# Comparing branches
git diff main summary-statistics

# Renaming a branch
git branch -m old_branch_name new_branch_name

# Deleting a branch
git branch -d branch_name

# Force-deleting a branch that hasn't been merged into main
# (the lowercase -d above will produce an error in that case)
git branch -D branch_name
```

## Merging Branches

Merging brings the changes from a **source** branch into a **destination** branch. Always switch to the destination first.

```bash
# First move to the destination branch
git switch main

# Merge a source branch into the current (destination) branch
git merge source
git merge ai-assistant

# Or specify both source and destination explicitly
git merge source destination
git merge ai-assistant main
```

# Collaborating with Git

## Merge Conflicts

A **conflict** is an inability to resolve differences in the contents of one or more files between branches. When Git cannot automatically merge changes, it marks the conflicting sections in the affected files and asks the user to resolve them manually.

```bash
# Open the conflicted file in a terminal editor to resolve the conflict
nano README.md
```

After editing the file to keep the desired changes, stage and commit it normally to complete the merge.

## Introduction to Remotes

Benefits of remote repos:

- Everything is backed up
- Collaboration, regardless of location

```bash
# Generic clone syntax
git clone path-to-project-repo

# Cloning a local project
git clone /home/george/repo

# Cloning a local project and giving the new copy a different name
git clone /home/george/repo new_repo

# Cloning from a remote repo
git clone URL

# Listing all remotes associated with the repo
git remote

# Listing remotes with their URLs
git remote -v
```

When cloning a repo, Git remembers where the original was by adding a **remote** entry (named `origin` by default) to the new repo's configuration.

## Pulling from Remotes

```bash
# Fetch from the origin remote (downloads changes but does not merge them)
git fetch origin

# Fetch a specific branch from the remote
git fetch origin main

# Fetch and merge from the remote's default branch into the local repo's current branch
git pull origin

# Pull from the origin remote's dev branch
git pull origin dev

# Compare local changes with the remote
git diff origin
```

`git fetch` only downloads changes from the remote, while `git pull` downloads **and** merges them into the current branch.

## Pushing to Remotes

```bash
# Generic push syntax
git push remote local_branch

# Push changes from the local main branch to the origin remote
git push origin main
```

## Skills Demonstrated

### Branching Workflows
- Creating, switching, renaming and deleting branches
- Diffing branches to understand their differences
- Merging branches into a destination branch

### Conflict Resolution
- Identifying and editing conflicted files
- Completing merges after manual resolution

### Remote Collaboration
- Cloning local and remote repositories
- Inspecting configured remotes
- Synchronizing changes with `fetch`, `pull` and `push`

## Key Takeaways

- **Branches let you work on parallel versions** of a project without affecting `main`.
- **Always switch to the destination branch before merging** — `git merge source` brings `source` into the current branch.
- **`git diff` works on branches too** — `git diff branchA branchB` shows the difference between them.
- **Use `-d` to delete a merged branch and `-D` to force-delete an unmerged one** — the uppercase flag bypasses Git's safety check, so use it deliberately.
- **Merge conflicts must be resolved by hand** — edit the file, stage it, and commit to complete the merge.
- **Remotes are just bookmarks** — `origin` is the default name Git uses for the remote a repo was cloned from.
- **`git fetch` downloads, `git pull` downloads and merges** — choose based on whether you want to review changes first.
- **`git push remote local_branch` is how local work becomes shared work** — without it, your commits stay on your machine.
