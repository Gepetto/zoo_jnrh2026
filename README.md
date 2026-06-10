# MPPI Solvers — Class Tutorial

In this tutorial you will implement three sampling-based trajectory optimizers (MPPI, MPPIElite, MPPICMA) and benchmark them on two MuJoCo environments: CartPole Swing-Up and the Unitree G1 humanoid.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — the only tool you need to install manually
- Python 3.12+ (uv will handle this automatically)
- A Linux machine with EGL support for MuJoCo rendering (the lab machines are set up for this)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd zoo_jnrh2026
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` virtual environment and installs all dependencies (JAX, MuJoCo, evosax, …) from the locked versions in `uv.lock`. No manual pip installs needed.

### 3. Open the notebook

**Option A — VS Code (recommended):**

1. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VS Code.
2. Open the repo folder in VS Code (`File → Open Folder`).
3. Open `class_notebook.ipynb`.
4. When the kernel picker appears, click **Select Another Kernel → Python Environments** and choose the `.venv` entry inside the repo folder. If it does not appear, press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and point it to `.venv/bin/python`.

**Option B — JupyterLab in the browser:**

```bash
uv run jupyter lab
```

Both options work — VS Code is recommended for its integrated debugger and file navigation.

## Repository structure

```
zoo-tuto/
├── class_notebook.ipynb   # Main notebook — work here
├── cartpole_task.py        # CartPole Swing-Up environment
├── humanoid_task.py        # Unitree G1 humanoid environment
├── model/
│   ├── cartpole/           # CartPole MuJoCo XML
│   └── g1/                 # Unitree G1 MuJoCo XML + assets
├── pyproject.toml          # Project dependencies
└── uv.lock                 # Locked dependency versions
```

## What you will implement

Open `class_notebook.ipynb` and follow the steps:

1. **Warm-up** — run the provided SimpleES example to understand the ask-tell loop
2. **Step 1** — implement `_ask` and `_tell` for MPPI, MPPIElite, and MPPICMA
3. **Step 2 & 4** — run your solvers on CartPole and the Humanoid, compare convergence

You will also need to implement `zero_order_hold` in both `cartpole_task.py` and `humanoid_task.py` — see the docstrings there for instructions.

## Rendering

MuJoCo rendering uses EGL (headless OpenGL). If you see a `MUJOCO_GL` error, make sure the environment variable is set correctly at the top of the notebook:

```python
import os
os.environ["MUJOCO_GL"] = "egl"   # Linux / lab machines
# os.environ["MUJOCO_GL"] = "glfw"  # macOS (if running locally)
```
