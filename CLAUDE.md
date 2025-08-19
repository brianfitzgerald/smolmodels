# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install uv and create virtual environment
pip install uv
uv venv
source .venv/bin/activate
uv sync --group torch
uv sync --no-build-isolation --group training
pyright --createstub transformers
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_parser.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format and lint code
ruff format .
ruff check .

# Type checking
pyright
```

### Data Generation and Synthetic Tasks
```bash
# Generate synthetic data using various tasks
python generate.py --task_name=roleplaying_game --model_name=gpt-4o-mini
python generate.py --task_name=gutenberg_backtranslation_from_txt --model_name=claude-3-5

# Download existing datasets
python util_scripts.py download_dataset gutenberg_backtranslate_from_txt
```

### Training Models
```bash
# Train models using TRL wrapper with predefined configurations
python train_trl.py --config=dolphin --wandb=True
python train_trl.py --config=creative_writing --notebook_mode=False
```

### Evaluation
```bash
# Run evaluations on models
python evaluate.py --model_name=gpt-4o-mini --eval_task=eq_bench_writing
```

### Modal (Cloud Computing)
```bash
# Training on Modal
uv run modal run -d modal_entrypoint.py::training

# Generation on Modal  
uv run modal run -d modal_entrypoint.py::generation

# Deploy inference endpoint
modal deploy modal_vllm.py
python util_scripts.py test_openai_api
```

## Architecture Overview

### Core Components

**Synthetic Data Generation Pipeline**: The system centers around generating high-quality synthetic datasets for training small language models. Key components include:

- **`synthetic_data/generation.py`**: Central generation system with multiple model wrappers (OpenAI, Anthropic, Gemini, OpenRouter, vLLM) supporting different providers and autoscaling
- **`synthetic_data/tasks/`**: Task-specific data generation modules inheriting from `BaseTask`, including writing tasks, roleplaying scenarios, and evaluation benchmarks
- **`generate.py`**: Main orchestration script for running synthetic data generation tasks

**Training Infrastructure**: Built on top of TRL (Transformers Reinforcement Learning) with custom wrappers:

- **`trl_wrapper/trainer_wrapper.py`**: Main training orchestrator with support for SFT, DPO, and GRPO training methods
- **`trl_wrapper/wrapper_config.py`**: Configuration system for different model architectures and training setups
- **Model definitions**: Located in `model/` with support for various architectures (CausalLM, T5, Vision)

**Evaluation System**: Comprehensive evaluation pipeline for model assessment:

- **`evaluation/code_execution.py`**: Code evaluation and execution framework
- **`synthetic_data/tasks/evals.py`**: Evaluation task implementations
- **`evaluate.py`**: Main evaluation runner with support for various benchmarks

### Key Design Patterns

**Task-Based Generation**: All synthetic data generation follows a common pattern:
1. Inherit from `BaseTask` in `synthetic_data/tasks/__init__.py`
2. Implement required methods: `format_input_conversation()`, `format_output_rows()`, `generate()`
3. Define dataset formats, columns, and generation parameters
4. Support for different generation models and autoscaling (see `RoleplayingGame` for advanced example)

**Multi-Model Generation**: The generation system supports using different models for different steps of the same task. For example, `RoleplayingGame` uses one model for initial scenario generation and a different (typically faster/cheaper) model for follow-up question generation, with independent autoscaling for each model.

**Configuration-Driven Training**: Training configurations are centralized in `CONFIGS` dictionary in `trl_wrapper/wrapper_config.py`, allowing easy experimentation with different model sizes, datasets, and hyperparameters.

**Modular Data Modules**: Dataset handling is abstracted through data modules in `dataset/` that handle loading, preprocessing, and batching for different data types (conversations, code contests, writing tasks).

### Model Support

**Supported Base Models**:
- SmolLM2 (135M) - Primary small model for experiments
- Llama 3.2 (1B, 3B), Llama 3.1 (8B)
- Mistral 7B, Ministral 8B
- Qwen 2.5 (0.5B, 1.5B, 3B), Qwen 3 (600M, 8B)

**Generation Models**: Supports 15+ remote models including GPT-4 variants, Claude models, Gemini, DeepSeek, and others through unified interface.

### Data Flow

1. **Seed Data**: Raw data from various sources (Project Gutenberg, screenplays, prompts)
2. **Task Processing**: Task-specific processing using generation models
3. **Output Formatting**: Standardized output format suitable for training
4. **Training Data**: Processed data fed into training pipeline
5. **Model Training**: TRL-based training with various optimization techniques
6. **Evaluation**: Comprehensive evaluation on various benchmarks

### Environment Integration

**Modal Integration**: Full cloud computing support for both training and inference through Modal platform integration.

**Development Modes**: Support for CLI, notebook, and Modal execution modes throughout the codebase.

**GPU/CPU Flexibility**: Training system automatically detects and configures for available hardware (CUDA, Metal, CPU).

### Key Files to Understand

- `synthetic_data/generation.py`: Generation wrapper system and model configurations
- `synthetic_data/tasks/__init__.py`: Base task interface that all generation tasks implement
- `trl_wrapper/trainer_wrapper.py`: Core training orchestration and configuration
- `generate.py`: Main CLI interface for data generation
- `train_trl.py`: Main CLI interface for model training
- `evaluate.py`: Main CLI interface for model evaluation