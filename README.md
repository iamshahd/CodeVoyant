# Code-Graph-Visualizer


## Quick Start
For Setup:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\Activate.ps1`
pip install uv
uv sync --active
```

For data, you may download our already preprocessed data from [google drive](https://drive.google.com/file/d/12gqk6P4MjqZ0aYGVGZ1_zJw-RGJDD9RK/view?usp=sharing) which includes graphs for 100 popular repositories from GitHub. These repositories were chosen for their purely Python codebases and popularity. You should extract the zip file inside `./codevoyant_output`. You may also analyze any repoository and generate graphs:
```bash
uv run analyze.py --repo <path_to_repo> --output <output_directory> 
```

To run the Streamlit app:
```bash
uv run streamlit run src/ui/app.py
```

To run benchmarking, you should make sure you download the data as mentioned above. Then run:
```bash
uv run python -m src.benchmark.run
```

## Development
After finishing the quick start, install development dependencies:
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
