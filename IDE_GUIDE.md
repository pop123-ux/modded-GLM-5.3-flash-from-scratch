# Read, run and extend the project

## Which notebooks need the repository?

| Notebook | Source needed? | Execution expectations |
| --- | --- | --- |
| `theory/*.ipynb` | No, for reading | Educational notes; most are Markdown. Vision code cells are illustrative snippets requiring surrounding model variables, not a standalone runnable pipeline. Planned lessons are explicitly marked. |
| `colab/modded_glm_experiments.ipynb` | Yes | Run inside Colab with the repository cloned into its runtime, or use the separate ZIP-upload fallback. The source-loading cell establishes the working directory. |
| `notebooks/local_quickstart.ipynb` | Yes | Run locally in a cloned checkout or extracted project using its Python environment; supports VS Code and Jupyter. |

Opening a notebook from GitHub in Colab does **not** clone the repository into the runtime. For experiments choose `SOURCE = 'github'`: the first code cell clones the repository, checks out REF, and switches to the cloned repo. Subsequent commands operate on that cloned repo, not a ZIP. A private repository needs authenticated access; the supplied unauthenticated clone option is for after publication. While private, use the clearly separate `SOURCE = 'upload'` fallback with `modded-glm-flash-from-scratch.zip`.

Theory notebooks can be read directly through [the lesson index](theory/README.md), GitHub, VS Code, Jupyter or Colab without cloning or installing dependencies. Do not use Run all on theory notes expecting a training experiment. Use the dedicated experiment notebooks instead.

## Local setup: cloned checkout

Clone through your normal authenticated Git client while private; the same command works without authentication once public:

```bash
git clone https://github.com/pop123-ux/modded-GLM-5.3-flash-from-scratch.git
cd modded-GLM-5.3-flash-from-scratch
```

Use Python 3.12. Create an environment from the repository root:

```bash
# Linux/macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For CPU-only Linux/Windows install the CPU wheel first. For another supported device use the repository's pinned torch with the appropriate wheel and driver; the Colab guide specifies its CUDA setup.

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-notebooks.txt
python -m pip install -e .
python -m pytest -q
```

An extracted archive works too: open the extracted folder containing `pyproject.toml`, `glm53_flash/` and `scripts/`, then run the same environment setup. That is an extracted project, not a cloned checkout; it has no Git history until you initialize one.

## VS Code

1. Install Microsoft's Python and Jupyter extensions (recommended in this workspace).
2. File → Open Folder: select the repository root, not only `theory/`.
3. Run **Python: Select Interpreter** and select `.venv`.
4. Open `notebooks/local_quickstart.ipynb`; use the notebook's **Select Kernel** picker to select the same `.venv` environment.
5. Run its cells in order for a tiny CPU training check. Read `theory/*.ipynb` as educational material.

The notebook kernel and the terminal interpreter can differ. Verify `sys.executable` in the notebook before running experiments. Other IDEs should also use `.venv` and the repository root as the working directory.

## JupyterLab or classic Notebook

With `.venv` activated, register a descriptive kernel and start the UI from the cloned repository root:

```bash
python -m ipykernel install --user --name modded-glm --display-name "Modded GLM (.venv)"
python -m jupyter lab
# Or: python -m jupyter notebook
```

Choose **Modded GLM (.venv)** for notebook execution. Open `notebooks/local_quickstart.ipynb`. The local notebook discovers the repository root from its working directory and uses the current kernel interpreter to launch scripts. It contains no Colab/Drive imports, `/content` paths or environment downloads.

## Continue researching

Use [COLAB.md](COLAB.md) for GPU campaigns and plots, [research/EXTENSIONS.md](research/EXTENSIONS.md) for controlled comparisons, and [CONTRIBUTING.md](CONTRIBUTING.md) for proposing new ideas. Keep outputs under `runs/`; use fresh output names and retain receipts. Local and Colab timing should only be compared with matched hardware/software settings.

References: [VS Code notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks), [IPython kernel setup](https://ipython.readthedocs.io/en/stable/install/kernel_install.html), [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
