# Contributing to spatialproteomics

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button at the top right of the repository page to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/your-username/spatialproteomics.git
cd spatialproteomics
```

## 3. Project Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management.
You can set up an environment as follows:

```
uv venv --python=python3.12  # specifying the python version is optional
source .venv/bin/activate  # activates the virtual environment
uv pip install -e ".[dev,all]"  # installs a local copy of the package
```

### 4. Make Your Changes

Create a new branch for your work:

```bash
git switch -C my-feature-branch
```

Make your changes, commit them with descriptive messages, and push the branch to your fork:

```bash
git add .
git commit -m "Describe your changes"
git push origin my-feature-branch
```

### 5. Using the Makefile

The project includes a `Makefile` to simplify common development tasks. To set up the environment, install dependencies, run tests, and perform linting in one step, you can use:

```bash
make all
```

### 6. Open a Pull Request

Once you are happy with your changes and all tests pass, go to the repository on GitHub and open a pull request from your branch.

## Code Style

- Follow [PEP8](https://pep8.org/) for Python code.
- Write clear, concise commit messages.
- Add or update tests for new features or bug fixes.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub and provide as much detail as possible.

---

Thank you for helping make **spatialproteomics** better!
