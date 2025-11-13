# UV Setup Guide for GeneAgent

This guide explains how to use `uv` to manage your GeneAgent environment with BaseAgent properly configured.

## ✅ Current Setup (Already Configured!)

Your `pyproject.toml` is already configured correctly:

```toml
[tool.uv.workspace]
members = [
    "BaseAgent",
]

[tool.uv.sources]
baseagent = { workspace = true }
```

This means BaseAgent is installed as a **workspace member** - it's treated as part of your project but maintained in its own directory.

---

## 🚀 Quick Start

### 1. Setup Environment (First Time)

```bash
# Create and sync environment
uv sync

# Activate the environment
source .venv/bin/activate
```

That's it! BaseAgent is automatically installed from `./BaseAgent` directory.

### 2. Run Your Scripts

```bash
# Make sure environment is activated
source .venv/bin/activate

# Run your scripts directly
python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o

# Or use uv run (no activation needed)
uv run python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o
```

### 3. Verify BaseAgent Import

```bash
# Test that BaseAgent is properly available
python -c "from BaseAgent import BaseAgent; print('✅ BaseAgent imported successfully!')"
python -c "from BaseAgent.llm import get_llm; print('✅ get_llm imported successfully!')"
```

---

## 📦 Managing Dependencies with UV

### Add New Packages

```bash
# Add a new package
uv add package-name

# Add with version constraint
uv add "package-name>=1.0.0"

# Add to optional dependencies
uv add --optional anthropic "langchain-anthropic>=0.1.0"
```

### Remove Packages

```bash
uv remove package-name
```

### Update Packages

```bash
# Update all packages
uv sync --upgrade

# Update specific package
uv sync --upgrade-package package-name
```

### Update BaseAgent

Since BaseAgent is a workspace member, just edit the code in `./BaseAgent` and it's immediately available:

```bash
# No need to reinstall! Changes are live.
cd BaseAgent
git pull  # If you want to update from git
```

---

## 🔧 Common UV Commands

### Environment Management

```bash
# Create/sync environment (first time or after changes)
uv sync

# Sync without updating lock file
uv sync --frozen

# Remove environment
rm -rf .venv
```

### Running Scripts

```bash
# With activation
source .venv/bin/activate
python script.py

# Without activation (recommended)
uv run python script.py
uv run python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o
```

### Check Installed Packages

```bash
uv pip list
uv pip show baseagent
```

---

## 🎯 Why UV Workspace?

### Advantages

1. **Editable Install**: Changes in `./BaseAgent` are immediately available
2. **No Separate pip install**: BaseAgent is managed by uv automatically
3. **Version Tracking**: `uv.lock` tracks exact versions for reproducibility
4. **Fast**: uv is much faster than pip
5. **Clean**: No confusion about where BaseAgent is installed from

### How It Works

```
GeneAgent/
├── BaseAgent/              # ← Workspace member (editable)
│   ├── BaseAgent/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   └── ...
│   └── pyproject.toml
├── llm_utils.py           # ← Can import from BaseAgent
├── worker.py              # ← Can import from BaseAgent
├── main_cascade.py        # ← Can import from BaseAgent
└── pyproject.toml         # ← Declares BaseAgent as workspace member
```

When you run `uv sync`, it:
1. Creates `.venv`
2. Installs all dependencies from `pyproject.toml`
3. Installs BaseAgent from `./BaseAgent` in editable mode
4. Locks versions in `uv.lock`

---

## 🐛 Troubleshooting

### Problem: Import Error "No module named 'BaseAgent'"

**Solution**:
```bash
# Resync environment
uv sync

# Verify installation
uv pip show baseagent
```

### Problem: Changes in BaseAgent Not Reflected

**Solution**: With workspace members, changes are immediate. If not working:
```bash
# Remove and recreate environment
rm -rf .venv
uv sync
```

### Problem: "pip" Points to Homebrew pip

This is **expected and fine** with uv! Don't use `pip` directly:

```bash
# ❌ Don't use pip
pip install package

# ✅ Use uv instead
uv add package
```

If you absolutely need pip:
```bash
source .venv/bin/activate
python -m ensurepip --upgrade
```

### Problem: "Multiple top-level packages" Error

This happens when trying `pip install -e .` **You don't need it!** Just use:

```bash
uv sync  # Already installs everything including BaseAgent
```

---

## 📚 Complete Workflow Example

### Initial Setup (First Time)

```bash
cd /Users/lib/GitHub/GeneAgent

# Create environment and install everything
uv sync

# Verify it works
uv run python -c "from BaseAgent import BaseAgent; print('✅ Success!')"
```

### Daily Usage

```bash
# Activate environment
source .venv/bin/activate

# Run your analysis
python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o

# Or without activation
uv run python main_cascade.py --input Datasets/MsigDB/MsigDB_toy.csv --llm gpt-4o
```

### Updating Dependencies

```bash
# Add new package
uv add new-package

# Update BaseAgent code
cd BaseAgent
git pull  # If it's a git repo
cd ..

# No need to reinstall - changes are live!
```

### Fresh Setup on New Machine

```bash
git clone <your-repo>
cd GeneAgent
uv sync  # Installs everything from uv.lock
```

---

## 🎓 UV vs PIP Comparison

| Task | pip | uv |
|------|-----|-----|
| Install packages | `pip install package` | `uv add package` |
| Remove packages | `pip uninstall package` | `uv remove package` |
| Update packages | `pip install --upgrade package` | `uv sync --upgrade` |
| Create venv | `python -m venv .venv` | `uv sync` |
| Install from pyproject.toml | `pip install -e .` | `uv sync` |
| Lock dependencies | `pip freeze > requirements.txt` | Automatic (`uv.lock`) |
| Editable install | `pip install -e ./BaseAgent` | Workspace member |

---

## ✅ Best Practices

1. **Always use uv commands** - Don't mix with pip
2. **Commit uv.lock** - For reproducibility
3. **Use `uv run`** - No need to activate venv
4. **Edit BaseAgent directly** - Changes are immediate
5. **Run `uv sync`** after pulling changes

---

## 🔗 Resources

- UV Documentation: https://docs.astral.sh/uv/
- Your pyproject.toml: Contains all configuration
- Your uv.lock: Contains exact versions

---

## Quick Reference Card

```bash
# Setup (first time)
uv sync

# Activate environment
source .venv/bin/activate

# Run scripts (no activation needed)
uv run python script.py

# Add/remove packages
uv add package-name
uv remove package-name

# Update everything
uv sync --upgrade

# Verify BaseAgent
python -c "from BaseAgent import BaseAgent"
```

---

**Current Status**: ✅ Your environment is properly configured!

Just run `uv sync` and you're ready to go!

