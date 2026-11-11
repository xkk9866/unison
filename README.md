# Unison

Unison is an experimental runtime for reusing work across concurrent multi-agent question answering requests. It combines agent orchestration, retrieval, execution policies, and a vLLM plugin for shared decoding.

## Repository layout

- `unison/`: runtime, agents, retrieval, cache, drafting, and execution policies.
- `unison_plugin/`: vLLM plugin package.
- `scripts/`: data preparation, training, experiments, and analysis utilities.

## Requirements

The implementation targets Linux with an NVIDIA GPU and a compatible vLLM installation. Python dependencies used by the scripts include PyTorch, NumPy, Transformers, and retrieval packages such as FAISS. The exact vLLM version and model weights must be supplied for the target environment; the plugin hooks into vLLM internals and may require adaptation across releases.

## Setup

1. Create a Python environment with the required GPU packages and vLLM.
2. Install the plugin: `pip install -e ./unison_plugin`.
3. Set `UNISON_ROOT` to a working directory containing `data/`, `models/`, and `results/`.
4. Update the model paths in `unison/common.py` for your local model weights.
5. Prepare datasets and indexes using the relevant scripts in `scripts/`.

## Running an experiment

For example, after preparing the data and model paths:

```bash
python scripts/run_exp.py --ds hotpotqa --split eval --methods noshare --n 200 --model qwen3-8b --tag baseline
```

Run `python scripts/run_exp.py --help` for available arguments. Several execution policies require trained checkpoints or precomputed replay data; the baseline command is the simplest starting point.

## Scope

This repository contains source code only. Datasets, model weights, checkpoints, generated results, and manuscript materials are not included.
