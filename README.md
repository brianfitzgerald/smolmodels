# Smolmodels

Collection of experiments related to tuning small language models for specific tasks.

## Installation

```bash
pip install uv
uv venv
uv sync --no-install-package flash-attn
uv sync --no-build-isolation
uv pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.44.2.dev0-py3-none-manylinux_2_24_x86_64.whl' --no-deps
```

### vLLM

```
export CUDA_HOME=/usr/local/cuda
sudo apt purge cmake
uv pip install setuptools_scm cmake
uv pip install vllm -vv --no-build-isolation
```


llama-cpp can also be installed with `CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python[server]`

### Pyright dependencies

```
pyright --createstub transformers
```