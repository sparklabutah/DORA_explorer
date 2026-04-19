<h1 align="center"> DORA: Directed exploration (MAB + TALE-Suite) </h1>

<p align="center">
  <a href="https://dora-explore.github.io/"><img src="https://img.shields.io/badge/Project%20Page-dora--explore.github.io-4285F4?style=flat&logo=homeassistant&logoColor=white&color=006A4E&labelColor=gray" alt="Project page"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg?logo=arxiv&labelColor=FFFFFF&logoColor=b31b1b" alt="arXiv"/></a>
  <a href="https://github.com/sparklabutah/DORA_explorer"><img src="https://img.shields.io/badge/GitHub-sparklabutah%2FDORA__explorer-blue?logo=GitHub&labelColor=black" alt="GitHub"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="license"/></a>
</p>

**tldr.** Research code for **DORA Explorer** (Diversity-Oriented Ranking of Actions): a **Bernoulli multi-armed bandit (MAB)** stack with LLM agents and classical baselines, and **custom agents** on top of [Microsoft TALE-Suite](https://github.com/microsoft/tale-suite) (text-adventure environments). DORA uses **token-level scoring** (mean log-probability and variance) plus a **λ schedule** or **auto-λ** to balance exploration and exploitation. **Project page:** [dora-explore.github.io](https://dora-explore.github.io/).

---

## Table of contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Multi-armed bandit (mab)](#multi-armed-bandit-mab)
- [TALE-Suite (tale-suite)](#tale-suite-tale-suite)
- [Analysis and figures](#analysis-and-figures)
- [Citation](#citation)

---

## Installation

**Python:** use **3.12+** at the repo root so `pip install -e tale-suite/` matches [`tale-suite/pyproject.toml`](tale-suite/pyproject.toml) (`requires-python >= 3.12`). The MAB code runs on 3.12 as well.

**Dependency files (kept separate):**

| File | Scope |
|------|--------|
| [`mab/requirements.txt`](mab/requirements.txt) | NumPy, PyTorch, Transformers, plotting, etc., for bandit experiments only. |
| [`tale-suite/requirements.txt`](tale-suite/requirements.txt) | Gymnasium, game stacks, `llm` plugins, **torch**, **tqdm**, etc., for TALE benchmarks (pulled in by `pip install -e tale-suite/`). |
| [`requirements.txt`](requirements.txt) (repo root) | Thin wrapper: `-r mab/requirements.txt` only. |

### Conda (optional)

[`environment.yml`](environment.yml) is **optional**: it pins **Python 3.12** and base packaging tools for a reproducible conda env on clusters (same idea as [TimeWarp](https://github.com/sparklabutah/timewarp)). If you prefer **venv + pip only**, skip conda and use the manual commands below.

With conda, from the repository root:

```sh
bash setup.sh
```

This creates the `dora-paper-code` environment, runs `pip install -e .` (MAB deps from [`pyproject.toml`](pyproject.toml)), then `pip install -e tale-suite/`.

⚠️ **GPU / models:** MAB LLM runs need a CUDA-capable machine and a [Hugging Face](https://huggingface.co/) token for gated models. TALE-Suite agents need API keys for your chosen LLM backend (`llm` package) plus HF access for the scoring model when using lambda agents. ⚠️

### Environment variables (MAB)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face Hub token (also accepts `HUGGING_FACE_HUB_TOKEN` in some tools). |
| `HF_MODEL` | Model id for generation in `mab/agents/llm.py` (default: `meta-llama/Llama-3.1-8B-Instruct`). |
| `SCORING_MODEL` | Optional override for the scoring model in `mab/score.py` (defaults to `HF_MODEL`). |

Place tokens in a `.env` file in the directory from which you run scripts, or export them in your shell.

### Manual install (venv / pip, no conda)

```sh
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r mab/requirements.txt          # MAB only
# Optional: editable root meta-package (same deps as mab/requirements.txt)
pip install -e .
# For TALE benchmarks:
pip install -e tale-suite/
```

---

## Repository layout

| Path | Role |
|------|------|
| [`mab/`](mab/) | Bandit environment, baselines, DORA policy, runners, plotting. |
| [`tale-suite/`](tale-suite/) | TALE-Suite fork with custom `agents/*.py` and `benchmark.py`. |
| [`setup.sh`](setup.sh) | Conda env + editable installs (same role as in [TimeWarp](https://github.com/sparklabutah/timewarp)). |
| [`environment.yml`](environment.yml) | **Optional** conda env (Python 3.12); skip if you use venv only. |
| [`pyproject.toml`](pyproject.toml) | Package metadata and MAB dependency pins. |

---

## Multi-armed bandit (mab)

### Quick start

From `mab/` after activating the environment:

```sh
cd mab

# classical baselines (UCB, TS, Greedy, ε-Greedy) — 1000 episodes, no GPU
python run.py baselines

# DORA lambda policy (generate → score → λ-softmax) — requires GPU
python run.py dora

# scheduled temperature decay (high → low) LLM agent — requires GPU
python run.py scheduled-temp

# fixed-temperature sweep for the zero-shot LLM agent — requires GPU
python run.py temperature-sweep
```

All sub-commands accept `--horizon`, `--replicates`, `--K`, `--delta`.
Agent-specific flags (e.g. `--alpha`, `--temp-start`, `--temperatures`) are shown by `python run.py <cmd> -h`.

### Module map

| File | Description |
|------|-------------|
| `bandit_env.py` | K-armed Bernoulli bandit. |
| `agents/baselines.py` | UCB, Thompson Sampling, Greedy, ε-Greedy. |
| `agents/llm.py` | Zero-shot LLM bandit agent (`LLMBanditAgent`) and `query_llm`. |
| `score.py` | Token-level scoring for candidate answer strings. |
| `agents/dora_lambda_schedule.py` | DORA: generate N candidates → filter → score → scheduled-λ softmax. |
| `agents/scheduled_temp.py` | Scheduled temperature over the horizon. |
| `prompts.py`, `evaluation.py` | Prompts (incl. candidate generation) and evaluation metrics. |
| `run.py` | Unified CLI: sub-commands `baselines`, `dora`, `scheduled-temp`, `temperature-sweep`. |

---

## TALE-Suite (tale-suite)

Upstream benchmark: [microsoft/tale-suite](https://github.com/microsoft/tale-suite). This tree includes additional registered agents, for example:

| CLI name | Module | Idea |
|----------|--------|------|
| `lambda-explore` | `agents/dora_lambda_schedule.py` | Candidate generation + scoring + scheduled λ sampling. |
| `lambda-autonomous` | `agents/dora_auto_explore.py` | GREEDY vs EXPLORE + model-chosen λ. |
| `scheduled-temp` | `agents/scheduled_temp_llm.py` | Exponential temperature schedule baseline. |

Example (after `pip install -e tale-suite/`):

```sh
cd tale-suite
python benchmark.py lambda-autonomous \
  --llm gpt-4o-mini \
  --scoring-model meta-llama/Llama-3.1-8B-Instruct \
  --conversation \
  --envs TWCookingLevel1 \
  --nb-steps 100
```

See [`tale-suite/README.md`](tale-suite/README.md) for the upstream documentation.

---

## Analysis and figures

Outputs go under `mab/logs/` by default (ignored by `.gitignore`). Commit figures separately or attach them to the paper supplement.

---

## Citation

If you use this code, please cite **your paper** (update the BibTeX when the arXiv / proceedings entry is available) and the **TALE-Suite** / environment papers you rely on.

### This work (placeholder)

```bibtex
@misc{dora2026placeholder,
  title        = {TITLE: Directed Exploration via Token-Level Scoring},
  author       = {YOUR AUTHORS},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

### TALE-Suite (check their README for the official citation)

```bibtex
% Add citation from https://github.com/microsoft/tale-suite
```

---

## Acknowledgments

The [**DORA Explorer**](https://dora-explore.github.io/) project page accompanies this repository. Layout and release ergonomics follow the style of [**TimeWarp**](https://github.com/sparklabutah/timewarp) (badges, `tldr.`, table of contents, `setup.sh` + `environment.yml`, and a `scripts/` helper directory). The interactive text benchmark builds on [**TALE-Suite**](https://github.com/microsoft/tale-suite).
