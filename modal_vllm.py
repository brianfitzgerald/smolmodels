import asyncio
import os
from typing import Optional
import modal
from vllm import AsyncEngineArgs, AsyncLLMEngine
from loguru import logger

from scripts.modal_definitons import (
    MODEL_WEIGHTS_VOLUME,
    MODELS_VOLUME_PATH,
    VLLM_IMAGE,
    format_timeout,
    app,
)
from vllm.usage.usage_lib import UsageContext


from vllm.entrypoints.openai.api_server import (
    init_app_state,
    build_app,
)
from vllm.utils import (
    FlexibleArgumentParser,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)


def get_checkpoint_dir(
    base_run_dir: str,
    model_id: Optional[str] = None,
    run: Optional[str] = None,
    steps: Optional[int] = None,
) -> str:
    checkpoint_dir = model_id
    logger.info(f"model id: {model_id}, run: {run}, steps: {steps}")
    if run:
        run_directory = os.path.join(base_run_dir, run)
        logger.info(f"run_directory: {run_directory}")
        if not os.path.exists(run_directory):
            raise ValueError(f"Run directory {run_directory} not found")
        checkpoints = os.listdir(run_directory)
        logger.info(f"checkpoints: {checkpoints}")
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


async def get_server(args, **uvicorn_kwargs):
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    fastapi_app = build_app(args)

    model_config = await engine.get_model_config()
    await init_app_state(engine, model_config, fastapi_app.state, args)

    return fastapi_app


@app.function(
    image=VLLM_IMAGE,
    gpu="l40s",
    secrets=[modal.Secret.from_name("smolmodels")],
    volumes={MODELS_VOLUME_PATH.as_posix(): MODEL_WEIGHTS_VOLUME},
    timeout=format_timeout(hours=6),
    container_idle_timeout=format_timeout(minutes=1),
    allow_concurrent_inputs=1,
)
@modal.asgi_app()
def serve():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)

    # don't actually parse cli args, just return the object
    args = parser.parse_args([])

    args.model = get_checkpoint_dir(
        os.path.join(MODELS_VOLUME_PATH.as_posix(), "runs"),
        # "meta-llama/Llama-3.2-3B-Instruct",
        None,
        "03-23-2-21-106352-llama-3.2-3b-instruct-txt_bt-txt-bt",
    )
    logger.info(f"args.model: {args.model}")
    validate_parsed_serve_args(args)

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        server = event_loop.run_until_complete(get_server(args))
    else:
        # When using single vLLM without engine_use_ray
        server = asyncio.run(get_server(args))

    logger.info("got server", server)

    return server
