<h1 align="center"> DORA: Directed exploration (MAB + TALE-Suite) </h1>

<p align="center">
  <a href="https://dora-explore.github.io/"><img src="https://img.shields.io/badge/Project%20Page-dora--explore.github.io-4285F4?style=flat&logo=homeassistant&logoColor=white&color=006A4E&labelColor=gray" alt="Project page"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-TBD-b31b1b.svg?logo=arxiv&labelColor=FFFFFF&logoColor=b31b1b" alt="arXiv"/></a>
  <a href="https://github.com/sparklabutah/DORA_explorer"><img src="https://img.shields.io/badge/GitHub-sparklabutah%2FDORA__explorer-blue?logo=GitHub&labelColor=black" alt="GitHub"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="license"/></a>
</p>

**TL;DR**  
Research code for **DORA Explorer** — an inference-time framework to improve LLM exploration.  
Includes:
- **MAB** (bandits + baselines + LLM agents)  
- **TALE-Suite agents** (text-based environments)

---

## Installation

> **Requirements:** Python **3.12+**, CUDA GPU for LLM agents, [Hugging Face](https://huggingface.co/) token for gated models.

### Option A — Conda (recommended)

```sh
bash setup.sh
```

Creates the `dora-paper-code` env, installs MAB deps and TALE-Suite in editable mode.

### Option B — pip / venv

```sh
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # MAB dependencies
pip install -e tale-suite/       # TALE-Suite dependencies
```

### Environment variables

Create a `.env` file or export these in your shell:

```sh
HF_TOKEN=<your-huggingface-token>
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct   # generation model (MAB)
SCORING_MODEL=                                # scoring model override (defaults to HF_MODEL)
```

TALE-Suite agents also need API keys for your chosen LLM backend (e.g. `OPENAI_API_KEY`).

---

## Repository layout

```
.
├── mab/                  # Bandit environment, baselines, DORA policy, runners
├── tale-suite/           # TALE-Suite fork with custom agents and benchmark.py
├── setup.sh              # Conda env + editable installs
├── environment.yml       # Optional conda env spec (Python 3.12)
└── pyproject.toml        # Package metadata and MAB dependency pins
```

---

## Multi-armed bandit (MAB)

### Quick start

```sh
cd mab

python run.py baselines          # UCB, TS, Greedy, epsilon-Greedy (no GPU)
python run.py dora               # DORA lambda policy (GPU)
python run.py scheduled-temp     # temperature schedule baseline (GPU)
python run.py temperature-sweep  # fixed-temperature sweep (GPU)
```

All sub-commands accept `--horizon`, `--replicates`, `--K`, `--delta`.
Run `python run.py <cmd> -h` for agent-specific flags.

### Module map

| File | Description |
|------|-------------|
| `bandit_env.py` | K-armed Bernoulli bandit environment. |
| `agents/baselines.py` | UCB, Thompson Sampling, Greedy, epsilon-Greedy. |
| `agents/llm.py` | Zero-shot LLM bandit agent and `query_llm`. |
| `agents/dora_lambda_schedule.py` | DORA: candidates -> score -> scheduled-lambda softmax. |
| `agents/scheduled_temp.py` | Scheduled temperature decay over horizon. |
| `score.py` | Token-level log-prob and variance scoring. |
| `prompts.py` | Prompt templates (incl. candidate generation). |
| `evaluation.py` | Evaluation metrics. |
| `run.py` | Unified CLI entry point. |

---

## TALE-Suite

Upstream: [microsoft/tale-suite](https://github.com/microsoft/tale-suite). This fork adds three exploration agents:

| CLI name | Module | Description |
|----------|--------|-------------|
| `dora-schedule` | `agents/dora_schedule.py` | Candidate generation + scoring + scheduled lambda sampling. |
| `dora-auto-explore` | `agents/dora_auto_explore.py` | GREEDY vs EXPLORE + model-chosen lambda. |
| `scheduled-temp` | `agents/scheduled_temp.py` | Exponential temperature schedule baseline. |

### Quick start

```sh
cd tale-suite

python benchmark.py dora-auto-explore \
  --llm gpt-4o-mini \
  --scoring-model meta-llama/Llama-3.1-8B-Instruct \
  --conversation \
  --envs TWCookingLevel1 \
  --nb-steps 100
```

See [`tale-suite/README.md`](tale-suite/README.md) for upstream docs and full environment list.

---

## Citation

If you find this repository useful please cite us.

```bibtex
@misc{dora2026placeholder,
  title        = {},
  author       = {AUTHORS},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/XXXX.XXXXX},
}
```

Please also cite the original TALES paper:

```bibtex
@article{cui2025tales,
  title   = {TALES: Text-Adventure Learning Environment Suite},
  author  = {Christopher Cui and Xingdi Yuan and Ziang Xiao and
             Prithviraj Ammanabrolu and Marc-Alexandre C\^ot\'e},
  journal = {arXiv preprint arXiv:2504.14128},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.14128}
}
```

