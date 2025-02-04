# Smolmodels

Collection of experiments related to tuning small language models for specific tasks.

## Installation

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync --group torch
uv sync --no-build-isolation --group training
```

### vLLM

```
export CUDA_HOME=/usr/local/cuda
sudo apt purge cmake
uv pip install setuptools_scm cmake
uv pip install vllm -vv --no-build-isolation
```


llama-cpp can also be installed with:
```bash
uv pip install "llama-cpp-python[server]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### Pyright dependencies

```bash
pyright --createstub transformers
```

### Modal

```bash
modal run modal_trl.py
```