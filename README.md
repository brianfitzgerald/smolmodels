# Smolmodels

Collection of experiments related to tuning small language models for specific tasks.

## Installation

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync --group torch
uv sync --no-build-isolation --group training
pyright --createstub transformers
```

### vLLM

CUDA:
```bash
export CUDA_HOME=/usr/local/cuda
sudo apt purge cmake
uv pip install setuptools_scm cmake
uv pip install vllm -vv --no-build-isolation
```

Mac:
```bash
uv pip install pip
pip install vllm==0.7.0 --use-deprecated=legacy-resolver
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
# Training
modal run -d modal_entrypoint.py::training --config gutenberg
# Generation
modal run -d modal_entrypoint.py::generation --task-name gutenberg_backtranslation
# Inference
modal serve modal_vllm.py
```