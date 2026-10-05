# Running ALINA with uv

[Back to README](README.md)

This guide sets up and runs the ALINA project with [uv](https://docs.astral.sh/uv/), a fast Python package and project manager from Astral. It covers macOS (using Homebrew) and Windows.

The project already has a standard `pyproject.toml`, so **uv works with no changes to the repo**. One command (`uv sync`) installs Python, creates the virtual environment, installs `numpy`, `opencv-python`, and `matplotlib`, and makes the `alina` command available.

---

<a id="install-uv"></a>

## 1. Install uv

<a id="macos-homebrew"></a>

### macOS (Homebrew)

```bash
brew update
brew install uv
uv --version
```

Upgrade later with `brew upgrade uv`.

If you also need Git to clone the repo, install it with Homebrew too: `brew install git`. You do **not** need to install Python with Homebrew, because uv downloads and manages its own Python builds (see step 2).

*Alternative (no Homebrew):* `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal.

<a id="windows"></a>

### Windows

Choose one of the following, in **PowerShell**:

```powershell
# Option A: official installer
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Option B: WinGet
winget install --id=astral-sh.uv -e

# Option C: Scoop
scoop install main/uv
```

Then **close and reopen PowerShell** so `uv` is on your `PATH`, and check it:

```powershell
uv --version
```

If you need Git: `winget install --id Git.Git -e`.

---

<a id="set-up-the-project"></a>

## 2. Set up the project

These commands are the same on macOS and Windows. Run them from the repo root (the folder containing `pyproject.toml`).

```bash
cd alina                 # or wherever you cloned the repo

uv python pin 3.11       # writes .python-version; uv downloads Python 3.11 if needed
uv sync                  # creates .venv/, resolves deps, writes uv.lock, installs ALINA (editable)
uv run alina --help      # confirms the CLI works
```

What `uv sync` does:

- Creates a project-local virtual environment in `.venv/`.
- Resolves dependencies and writes **`uv.lock`**, which pins exact versions for every platform. Commit it so everyone gets identical versions.
- Installs ALINA in **editable mode**, so code edits in `alina/` or `eval/` take effect without reinstalling.
- Adds the `alina` command, defined in `[project.scripts]` in `pyproject.toml`.
- **Side effect:** creates an `alina.egg-info/` folder in the repo root. It's package metadata written by setuptools for the editable install: name, version, dependencies, and the `alina = cli:main` command. It contains no code or data. It's regenerated automatically, already excluded by `.gitignore`, and safe to delete (`rm -rf alina.egg-info && uv sync` brings it back).

The project supports Python 3.9 or newer (`requires-python = ">=3.9"`). Version 3.11 is a safe default; `ALINA_README.md` says it was tested on 3.10.

The repo's `.gitignore` already excludes `.venv/`.

---

<a id="run-alina"></a>

## 3. Run ALINA

Prefix commands with **`uv run`**. It runs them inside `.venv` and re-syncs automatically if `pyproject.toml` or `uv.lock` changed, so you never need to activate the environment.

<a id="check-the-evaluation"></a>

### Check the evaluation (no GUI needed)

macOS:

```bash
uv run alina evaluate \
  --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_1 data/gt_alina_labels/canny_textfiles/canny_textfiles_2 data/gt_alina_labels/canny_textfiles/canny_textfiles_3 \
  --alina-dirs data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_1 data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_2 data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_3
```

Windows (PowerShell uses a backtick `` ` `` for line continuation):

```powershell
uv run alina evaluate `
  --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_1 data/gt_alina_labels/canny_textfiles/canny_textfiles_2 data/gt_alina_labels/canny_textfiles/canny_textfiles_3 `
  --alina-dirs data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_1 data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_2 data/gt_alina_labels/ALINA_textfiles/ALINA_textfiles_3
```

Expected output, which matches the paper's 98.45% detection rate:

```
Average Recall: 98.44245354113903%
Average Precision: 92.29333214425482%
Average F1: 95.16844076391271%
```

<a id="label-frames"></a>

### Label frames (opens OpenCV windows; needs a desktop session)

Try one frame first to tune parameters:

```bash
uv run alina label --image data/Raw_Data/vidd_2/00001.jpg --output-images-dir outputs/debug --output-coords-dir outputs/debug --show
```

Then label a whole video folder:

```bash
uv run alina label --input-dir data/Raw_Data/vidd_2 --output-images-dir outputs/vidd_2/annotated --output-coords-dir outputs/vidd_2/coords --log-file outputs/vidd_2/timing.log
```

Click the ROI corners in the order **Bottom-Left, Top-Left, Top-Right, Bottom-Right**, press any key, then confirm the preview. The other subcommands (`video-to-frames`, `rotate-frames`, `resize-images`, `cbem`, `superimpose`) work the same way with `uv run alina <subcommand> ...`; see `ALINA_README.md` for their flags.

<a id="activate-the-environment"></a>

### Optional: activate the environment the classic way

```bash
source .venv/bin/activate        # macOS
.venv\Scripts\activate           # Windows PowerShell
alina --help
```

<a id="useful-uv-commands"></a>

### Useful uv commands

| Task | Command |
|---|---|
| Add a dependency (updates `pyproject.toml` and `uv.lock`) | `uv add scikit-image` |
| Add a dev-only tool | `uv add --dev pytest` |
| Remove a dependency | `uv remove scikit-image` |
| Upgrade all locked versions | `uv lock --upgrade && uv sync` |
| Run a one-off tool without installing it | `uvx ruff check .` |
| Use the old requirements file instead of the project | `uv venv && uv pip install -r requirements.txt` |
| Rebuild the environment from scratch | `uv sync --reinstall` (or delete `.venv/` and run `uv sync`) |
| Switch Python version | `uv python pin 3.12 && uv sync` |

---

<a id="why-uv"></a>

## 4. Why uv? Comparison with other options

| | **uv** | pip + venv | conda | Poetry | Pipenv |
|---|---|---|---|---|---|
| Install speed | Very fast (written in Rust; Astral benchmarks report 10–100× faster than pip) | Slow to moderate | Slow (dependency solving) | Moderate | Slow, especially locking |
| Installs Python itself | **Yes** (`uv python install`, `pin`) | No (needs python.org, pyenv, or Homebrew) | Yes | No | No (can call pyenv) |
| Lockfile for reproducible installs | **Yes**, `uv.lock` (one lockfile for all platforms) | No (only `pip freeze`) | Optional (`conda-lock`) | Yes, `poetry.lock` | Yes, `Pipfile.lock` |
| Works with this repo's `pyproject.toml` as-is | **Yes** | Yes (`pip install -e .`) | Needs a separate `environment.yml` | Yes in Poetry 2.x | No (uses `Pipfile`) |
| Creates and manages the virtual environment | Automatic (`.venv/`) | Manual (`python -m venv`, activate) | Automatic (named environments) | Automatic | Automatic |
| Run without activating | `uv run` (also auto-syncs) | No | `conda run` | `poetry run` | `pipenv run` |
| Run CLI tools without installing | `uvx` | No (needs `pipx`) | No | No | No |
| Disk usage across projects | Low (global cache with hard links) | High (full copy per environment) | High | Moderate | Moderate |
| Non-Python libraries (CUDA, GDAL, MKL) | No | No | **Yes**, its main strength | No | No |
| Installing the tool | One binary, no Python needed (`brew install uv`) | Comes with Python | Large installer (Miniconda or Miniforge) | Needs Python (`pipx install poetry`) | Needs Python |

<a id="why-uv-suits-alina"></a>

### Why uv suits ALINA in particular

- **One command from clone to running.** `uv sync` replaces about four manual steps: install the right Python, create a venv, activate it, and run `pip install -e .`.
- **Same steps on macOS and Windows.** Only the uv install command differs. Everything after that, including the `.venv` location and `uv run`, is identical, which makes classroom or team instructions simple.
- **Reproducible results.** `uv.lock` pins exact NumPy and OpenCV versions, so detection numbers don't drift because someone installed a newer OpenCV.
- **No conda needed.** ALINA's three dependencies all ship as prebuilt pip wheels, including OpenCV's GUI and video support, so conda's ability to install non-Python libraries adds nothing here.
- **No repo changes.** It reads the existing standard (PEP 621) `pyproject.toml`; the project doesn't need to adopt a `Pipfile`, `environment.yml`, or `[tool.poetry]` section.

<a id="when-another-tool"></a>

### When another tool might be better

- **conda / mamba:** if you later add GPU deep-learning or geospatial packages that need system libraries such as CUDA or GDAL.
- **pip + venv:** if you can't install any extra tools, since pip and venv come with Python.
- **Poetry:** if your team already uses its publishing workflow; uv can also build and publish packages (`uv build`, `uv publish`).

---

<a id="troubleshooting"></a>

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` after installing | Open a new terminal. On macOS check `brew --prefix`/bin is on your `PATH`; on Windows reopen PowerShell. |
| PowerShell refuses to run the installer script | Use the `-ExecutionPolicy ByPass` form shown above, or install with WinGet. |
| `.venv\Scripts\activate` is blocked on Windows | Skip activation and use `uv run ...`, or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| OpenCV windows don't appear | Run from a normal desktop terminal, not over SSH, in a container, or headless. `alina evaluate` is the only fully headless subcommand. |
| Matplotlib warns that its config directory isn't writable | Set `MPLCONFIGDIR` to a writable folder, e.g. `export MPLCONFIGDIR=/tmp/mpl` (macOS). |
| Wrong Python version in use | `uv python pin 3.11 && uv sync`, then check with `uv run python --version`. |
| Environment seems broken | `uv sync --reinstall`, or delete `.venv/` and run `uv sync`. |

---

<a id="verified"></a>

## Verified

Tested on macOS with uv 0.9.26 and Python 3.11.14, using a copy of this repository:

- `uv python pin 3.11` and `uv sync` resolved and installed the dependencies (OpenCV 5.0.0, NumPy 2.4.6, Matplotlib 3.11.2) and wrote `uv.lock`.
- `uv run alina --help` listed all seven subcommands.
- `uv run alina evaluate` on `data/gt_alina_labels/` printed the results shown above.
- `process_image()` ran end-to-end on a raw frame from Python without the GUI.

Not tested here: the interactive OpenCV windows (`label`, `cbem`, `superimpose`) and the Windows commands, which follow uv's official documentation.
