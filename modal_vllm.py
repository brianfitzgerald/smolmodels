import os
import fire
from typing import Optional
import modal

from scripts.modal_definitons import (
    MODEL_WEIGHTS_VOLUME,
    MODELS_VOLUME_PATH,
    VLLM_IMAGE,
    format_timeout,
    app,
)


def get_checkpoint_dir(
    base_run_dir: str,
    model: Optional[str] = None,
    run: Optional[str] = None,
    steps: Optional[int] = None,
) -> str:
    checkpoint_dir = model
    print(f"model: {model}, run: {run}, steps: {steps}")
    if run:
        run_directory = os.path.join(base_run_dir, run)
        print(f"run_directory: {run_directory}")
        if not os.path.exists(run_directory):
            raise ValueError(f"Run directory {run_directory} not found")
        checkpoints = os.listdir(run_directory)
        print(f"checkpoints: {checkpoints}")
        checkpoints = [x for x in checkpoints if x.startswith("checkpoint-")]
        sorted_checkpoints = list(
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        )
        sorted_checkpoints = [int(x.split("-")[-1]) for x in sorted_checkpoints]
        if steps in sorted_checkpoints:
            checkpoint_dir = f"{run_directory}/checkpoint-{steps}"
        elif steps is not None and steps not in sorted_checkpoints:
            raise ValueError(f"Checkpoint {steps} not found in {run_directory}")
        else:
            latest_ckpt = sorted_checkpoints[-1]
            checkpoint_dir = f"{run_directory}/checkpoint-{latest_ckpt}"
    assert checkpoint_dir is not None
    return checkpoint_dir


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config


TOKEN = "brianf"


@app.function(
    image=VLLM_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=6),
)
def serve(
    model: Optional[str] = None,
    run: Optional[str] = None,
    steps: Optional[int] = None,
):
    """
    base_run_dir is the directory containing the runs.
    If model is provided, use that ckpt.
    If run is provided, use the run specified by the run_name.
    If steps is provided, use that ckpt, otherwise use latest.
    """
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server  # type: ignore
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
    from vllm.entrypoints.logger import RequestLogger  # type: ignore
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat  # type: ignore
    from vllm.entrypoints.openai.serving_completion import (  # type: ignore
        OpenAIServingCompletion,
    )
    from vllm.entrypoints.openai.serving_models import BaseModelPath  # type: ignore
    from vllm.usage.usage_lib import UsageContext  # type: ignore

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title="OpenAI-compatible vLLM server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,  # type: ignore
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    router = fastapi.APIRouter()

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    checkpoint_dir = get_checkpoint_dir(
        os.path.join(MODELS_VOLUME_PATH.as_posix(), "runs"), model, run, steps
    )

    engine_args = AsyncEngineArgs(
        model=checkpoint_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)

    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [BaseModelPath(name="default", model_path=checkpoint_dir)]

    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        chat_template=None,
        response_role="assistant",
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )
    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    return web_app
