import asyncio
import os
from typing import Optional
import modal

from scripts.modal_definitons import (
    MODEL_WEIGHTS_VOLUME,
    MODELS_VOLUME_PATH,
    VLLM_IMAGE,
    format_timeout,
    app,
)

import uvloop
from vllm.entrypoints.openai.api_server import (
    run_server,
    init_app_state,
    create_server_socket,
    build_async_engine_client,
    build_app,
)
from vllm.utils import (
    FlexibleArgumentParser,
    set_ulimit,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)

from scripts.run_vllm import TIMEOUT_KEEP_ALIVE


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


async def get_server(args, **uvicorn_kwargs):
    async with build_async_engine_client(args) as engine_client:
        fastapi_app = build_app(args)

        model_config = await engine_client.get_model_config()
        await init_app_state(engine_client, model_config, fastapi_app.state, args)

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
    """
    base_run_dir is the directory containing the runs.
    If model is provided, use that ckpt.
    If run is provided, use the run specified by the run_name.
    If steps is provided, use that ckpt, otherwise use latest.
    """

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    # don't actually parse cli args, just return the object
    args = parser.parse_args([])
    args.model = get_checkpoint_dir(
        os.path.join(MODELS_VOLUME_PATH.as_posix(), "runs"),
        "meta-llama/Llama-3.2-3B-Instruct",
        None,
    )
    # HACK
    print(f"args.model: {args.model}")
    args.model = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"args.model: {args.model}")
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

    return server
