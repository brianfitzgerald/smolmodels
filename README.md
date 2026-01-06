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
uv run modal run -d  modal_entrypoint.py::training
# Generation
uv run modal run -d modal_entrypoint.py::generation
# Inference
modal deploy modal_vllm.py
python util_scripts.py test_openai_api
```

### Generation

```bash
uv run generate.py --task_name roleplaying_game --batch_size 1
```

# Utils
```bash
python util_scripts.py download_dataset gutenberg_backtranslate_from_txt
```