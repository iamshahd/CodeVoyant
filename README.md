# Code-Graph-Visualizer


## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\Activate.ps1`
pip install uv
uv sync --active
```


## Development
Same as quick start but
```bash
uv pip install ruff mypy types-networkx
```

Add extensions:
1. [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
2. [MyPy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)


You'll have format on save and type checking based on .vscode/settings.json.

You can specifically run mypy and ruff via:
```bash
mypy .
ruff format .
```
