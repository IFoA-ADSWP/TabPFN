# Simple Git & GitHub Guide with AI

**For TabPFN colleagues** — practical instructions for working with Git and AI assistance.

---

## Quick Start (5 minutes)

### 1. Install GitHub CLI

```bash
# macOS
brew install gh

# Authenticate
gh auth login
```

### 2. Verify It Works

```bash
# Check you're logged in
gh auth status

# List open issues
gh issue list

# List open PRs
gh pr list
```

---

## Daily Workflow

### Start Your Day

```bash
# Get latest changes
git pull origin main

# See what's happening
gh issue list --label "help wanted"
gh pr list --state open
```

### Work on a Task

```bash
# 1. Create a branch
git checkout -b fix/my-bug-fix

# 2. Make changes, then commit
git add .
git commit -m "fix: describe what you fixed"

# 3. Push and create PR
git push -u origin fix/my-bug-fix
gh pr create --title "Fix: describe the fix" --body "Closes #42"
```

### Review Someone Else's PR

```bash
# Check out their branch
gh pr checkout 123

# Or view on web
gh pr view 123 --web
```

---

## AI Can Help You With

### Using AI in Your IDE

If you're using **Cursor**, **Windsurf**, or **VS Code with Copilot**:

1. **Ask questions about the code**
   ```
   What does this function do?
   How does the data loading work in this notebook?
   ```

2. **Get help with Git**
   ```
   Create a commit message for these changes
   Help me resolve this merge conflict
   ```

3. **Write code**
   ```
   Write a function to load the eudirectlapse dataset
   Add error handling to this script
   ```

4. **Debug issues**
   ```
   Why is this test failing?
   Help me fix this import error
   ```

### Using AI via Command Line (if you have opencode)

```bash
# Ask a question
opencode "what does the bemtl97 dataset contain?"

# Get code help
opencode "write a script to compare TabPFN vs GLM on the Spanish motor dataset"
```

---

## Common Git Commands

### Branch Management

```bash
# List branches
git branch

# Switch to a branch
git checkout branch-name

# Create and switch to new branch
git checkout -b new-branch-name

# Delete a branch (after merge)
git branch -d branch-name
```

### Staging & Committing

```bash
# See what's changed
git status

# See detailed changes
git diff

# Stage specific files
git add file1.py file2.py

# Stage all changes
git add .

# Commit with a message
git commit -m "feat: add new dataset loader"

# Push to GitHub
git push origin branch-name
```

### Keeping Up to Date

```bash
# Get latest from main
git pull origin main

# If you have merge conflicts
git merge main
# Resolve conflicts in your editor, then:
git add .
git commit -m "merge: resolve conflicts"
```

### Undoing Mistakes

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes) — BE CAREFUL
git reset --hard HEAD~1

# See commit history
git log --oneline -10
```

---

## GitHub Commands (gh CLI)

### Issues

```bash
# List open issues
gh issue list

# List issues with a label
gh issue list --label "bug"

# View an issue
gh issue view 42

# Create an issue
gh issue create --title "Bug: something broke" --body "Steps to reproduce..."

# Close an issue
gh issue close 42
```

### Pull Requests

```bash
# List open PRs
gh pr list

# View a PR
gh pr view 123

# Create a PR
gh pr create --title "Fix: describe fix" --body "Closes #42"

# Check out a PR locally
gh pr checkout 123

# Merge a PR (if you have permission)
gh pr merge 123

# View PR in browser
gh pr view 123 --web
```

### Repository Info

```bash
# View repo info
gh repo view

# Clone a repo
gh repo clone IFoA-ADSWP/TabPFN

# Fork a repo
gh repo fork owner/repo
```

---

## AI Prompts That Work Well

### For Understanding Code

```
Explain how the TabPFN benchmark works in this repo
What's the difference between the classification and regression results?
How do I run the Spanish motor frequency analysis?
```

### For Writing Code

```
Write a Python script to load the eudirectlapse dataset and train a TabPFN classifier
Create a function to calculate AUC with confidence intervals
Help me add a new dataset to the benchmark suite
```

### For Debugging

```
Why am I getting "ModuleNotFoundError: No module named 'tabpfn'"?
The notebook is failing at cell 5 — what's wrong?
Help me fix this import error
```

### For Git Help

```
Write a good commit message for these changes
Help me resolve this merge conflict
What's the best way to rebase my branch onto main?
```

---

## Tips for Working with AI

### Be Specific

❌ **Bad:** "Fix this"
✅ **Good:** "The test in `tests/test_benchmark.py` is failing because the dataset path is wrong. Can you fix it?"

### Provide Context

❌ **Bad:** "Write a function"
✅ **Good:** "Write a Python function that loads the `eudirectlapse.csv` dataset, splits it into train/test, and returns the features and target. The target column is `lapse`."

### Ask for Explanation

```
Explain what this code does step by step
Why did you choose this approach?
What are the trade-offs of this solution?
```

### Verify AI Output

- **Always test code** before committing
- **Read the diff** before pushing
- **Ask questions** if something doesn't make sense

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Get latest | `git pull origin main` |
| See changes | `git status` |
| Stage files | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push origin branch-name` |
| Create PR | `gh pr create --title "..." --body "..."` |
| List issues | `gh issue list` |
| List PRs | `gh pr list` |
| Check out PR | `gh pr checkout 123` |
| Merge PR | `gh pr merge 123` |

---

## Getting Help

### From AI
```
I'm new to Git. Can you walk me through making my first commit?
```

### From Colleagues
- Ask in Slack/Teams
- Pair program with someone experienced
- Review others' PRs to learn patterns

### From Documentation
- Git docs: https://git-scm.com/doc
- GitHub docs: https://docs.github.com
- GitHub CLI docs: https://cli.github.com

---

## Common Mistakes (and How to Fix Them)

### "I committed to main by mistake"
```bash
# Undo the last commit, keep changes
git reset --soft HEAD~1

# Create a new branch
git checkout -b my-feature

# Re-commit
git add .
git commit -m "feat: my changes"
```

### "I have merge conflicts"
```bash
# Get the latest main
git pull origin main

# Fix conflicts in your editor (look for <<<<<<< markers)

# Stage the fixed files
git add .

# Complete the merge
git commit -m "merge: resolve conflicts"
```

### "I pushed something I shouldn't have"
```bash
# If it's the last commit
git reset --hard HEAD~1
git push --force-with-lease origin branch-name

# If it contains secrets, contact the repo admin immediately
```

---

*Last updated: 2026-08-20*
