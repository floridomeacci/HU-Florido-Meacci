
florido jan meacci

## Git Commands

```sh
# Initialize a new repository
git init
# Check status
git status
# Add files to staging
git add <file>
# Commit changes
git commit -m "commit message"
# View commit history
git log --oneline --graph
# Create a new branch
git checkout -b <branchname>
# Switch branches
git checkout <branchname>
# Merge a branch into current
git merge <branchname>
# Push to remote
git push -u origin <branchname>
# Pull from remote
git pull

## Regex Cheat Sheet

| Pattern | Description |
|---------|-------------|
| .       | Any character except newline |
| ^       | Start of string |
| $       | End of string |
| *       | 0 or more repetitions |
| +       | 1 or more repetitions |
| ?       | 0 or 1 repetition (or non-greedy) |
| []      | Any character in set |
| [^ ]    | Any character NOT in set |
| ( )     | Grouping |
| |       | OR |
| {n}     | Exactly n repetitions |
| {n,}    | n or more repetitions |
| {n,m}   | Between n and m repetitions |
| \\d      | Digit (0-9) |
| \\D      | Not a digit |
| \\w      | Word character (a-z, A-Z, 0-9, _) |
| \\W      | Not a word character |
| \\s      | Whitespace |
| \\S      | Not whitespace |
| \\b      | Word boundary |
| \\B      | Not a word boundary |

**Examples:**

```python
# Match a 3-digit number
r"\\d{3}"
# Match an email address
r"[\w.-]+@[\w.-]+\\.\w+"
# Match a line starting with 'Hello'
r"^Hello"
```

```

## Python Virtual Environment (venv)

```sh
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment (macOS/Linux)
source venv/bin/activate
# Activate the virtual environment (Windows)
venv\Scripts\activate
# Deactivate the virtual environment
deactivate
# Install packages
pip install <package>
```

