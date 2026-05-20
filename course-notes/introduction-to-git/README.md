# Introduction to Git

## Course Overview
An introductory course covering the fundamentals of version control with Git. The course walks through creating repositories, staging and committing changes, inspecting version history, comparing file versions, and restoring or reverting files — all from the shell.

## Key Topics Covered

### 1. Introduction to Version Control
- What version control is and why it matters
- Using Git from the shell / terminal
- Useful terminal commands (`pwd`, `ls`, `cd`)
- Checking the installed Git version

### 2. Creating Repos
- Git repository structure and the `.git` directory
- Benefits of using repositories
- Initializing a new repository with `git init`

### 3. Staging and Committing Files
- The Git workflow: edit → stage → commit
- Adding files to the staging area with `git add`
- Capturing snapshots with `git commit`

### 4. Version History
- Anatomy of a Git commit (commit, tree, blob)
- Git hashes and their role in tracking
- Viewing history with `git log`
- Filtering history by count, file, and date

### 5. Comparing Versions
- Comparing working directory with staging area using `git diff`
- Comparing staged changes against the last commit
- Comparing two commits using hashes or `HEAD~n`

### 6. Restoring and Reverting Files
- Reverting commits with `git revert`
- Restoring a single file from an earlier commit with `git checkout`
- Unstaging files with `git restore --staged`

## Key Concepts

### Initializing a repository
```bash
git init mental-health-workspace
```

### Staging and committing
```bash
git add README.md
git add .
git commit -m "Adding a README."
```

### Inspecting history
```bash
git log
git log -3
git log report.md
git log --since='Apr 2 2024' --until='Apr 11 2024'
git show c27fa856
```

### Comparing versions
```bash
git diff report.md
git diff --staged report.md
git diff 35f4b4d 126398f
git diff HEAD~1 HEAD
```

### Reverting and restoring
```bash
git revert --no-edit HEAD
git checkout HEAD~1 -- report.md
git restore --staged summary_statistics.csv
```

## Course Notes

# Introduction to Version Control

Version control is a method where the user can manage changes of files, programs and directories. With version control, the user can track files in different states, combine different versions of files, identify a particular version of a file and revert changes.

## Using Git

Git commands are run on the **shell**, also known as the **terminal**. The shell is a program for executing commands. It can also be used to easily preview or inspect files and directories (folders).

## Useful Terminal Commands

- `pwd` → **print working directory**. Shows the current working directory.
- `ls` → lists what is in the current directory. Shows files in a directory.
- `cd directory` → **change directory**. Navigates into the given directory. Using `cd ..` goes back to the previous directory.

```bash
# How to see git version
git --version
```

# Creating Repos

A **Git repository (repo)** is a directory containing files and sub-directories. Git stores all of its tracking information inside a hidden `.git` folder. The user should **not** edit anything inside `.git`.

## Benefits of Repos

- Systematically track versions
- Revert to previous versions
- Compare versions at different points in time
- Collaborate with colleagues

## Creating a New Repo

```bash
git init mental-health-workspace
```

This creates a new directory called `mental-health-workspace` and initializes a Git repository inside it.

# Staging and Committing Files

The typical Git workflow has three steps:

1. **Edit and save** files on the computer.
2. **Add the file(s) to the staging area** — tells Git what has been modified and is ready to be saved.
3. **Commit the files** — Git takes a snapshot of the staged files at that point in time, allowing comparisons and reverts later.

## Adding to the Staging Area

```bash
# Adding a single file
git add README.md

# Adding all modified files
git add .

# Committing the staged files with a descriptive message
git commit -m "Adding a README."
```

# Version History

## Viewing Version History

Each Git commit has three parts:

1. **Commit** — contains the metadata: author, log message, commit time.
2. **Tree** — tracks the names and locations of files and directories in the repo. Like a dictionary that maps keys to files/directories.
3. **Blob** — Binary Large OBject. May contain data of any kind. A compressed snapshot of a file's contents.

### Git Hash

Git uses a pseudo-random number generator — a **hash function** — to identify commits, trees and blobs.

Hashes allow data sharing between repos. If two files are the same, then their hashes are the same. Git only needs to compare hashes to detect identical content.

```bash
# Display commits in reverse chronological order (most recent first)
git log
```

## Version History Tips and Tricks

```bash
# Show the 3 most recent commits
git log -3

# Check the commit history of a single file
git log report.md

# Combine the two
git log -3 report.md

# Restrict git log by date
git log --since='Month Day Year'
git log --since='Apr 2 2024'

# Commits between 2nd and 11th April
git log --since='Apr 2 2024' --until='Apr 11 2024'

# Commits made yesterday
git log --since='yesterday'

# Display details of a particular commit by hash
git show c27fa856
```

## Comparing Versions

`git diff` shows the difference between versions of a file.

```bash
# Compare the working directory with the last committed version
git diff report.md

# Stage a file
git add report.md

# Compare the last committed version with the version in the staging area
git diff --staged report.md

# Compare two commits (most recent hash goes second)
git diff 35f4b4d 126398f

# HEAD refers to the latest commit. HEAD~1 is the commit before it.
git diff HEAD~1 HEAD

# Compare the staged version of a file against the previous commit
git diff --staged mental_health_survey.csv HEAD~1 HEAD
```

## Restoring and Reverting Files

```bash
# Reinstate a previous version and create a new commit
git revert <hash>

# Avoid opening the text editor for the revert commit message
git revert --no-edit HEAD

# Revert without committing
git revert -n HEAD

# Restore a single file from a previous commit
git checkout HEAD~1 -- report.md

# Unstage a single file
git restore --staged summary_statistics.csv

# Unstage all files
git restore --staged
```

## Skills Demonstrated

### Version Control Fundamentals
- Initializing and structuring Git repositories
- Tracking files through the stage/commit workflow
- Reading and interpreting commit history

### Shell Proficiency
- Navigating directories with `pwd`, `ls`, `cd`
- Running Git commands directly from the terminal

### File Comparison and Recovery
- Diffing working, staged and committed versions
- Restoring files from earlier commits
- Reverting unwanted changes safely

## Key Takeaways

- **Version control lets you track, compare and revert changes** to any file in a project.
- **A repo is just a directory** that Git tracks via its hidden `.git` folder — never edit `.git` by hand.
- **The Git workflow is edit → stage → commit** — commits are snapshots, not diffs.
- **Each commit has a unique hash** that identifies its content; identical content produces identical hashes.
- **`git log` is your time machine** — combine flags like `-n`, file paths and `--since` / `--until` to narrow down history.
- **`git diff` answers "what changed?"** between the working directory, the staging area, and any two commits.
- **`HEAD` is the latest commit**, and `HEAD~n` walks back `n` commits — most navigation commands accept either hashes or `HEAD~n`.
- **Prefer `git revert` over destructive edits** — it creates a new commit that undoes a previous one, preserving full history.
