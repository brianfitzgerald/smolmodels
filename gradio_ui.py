from typing import Iterable, Optional
import gradio as gr
from openai import OpenAI
import fire
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from loguru import logger
from huggingface_hub import hf_hub_download
import json


def _get_eos_tokens():
    config_path = hf_hub_download(
        repo_id="meta-llama/Llama-3.2-3B-Instruct", filename="generation_config.json"
    )
    config_json: dict = json.load(open(config_path))
    stop_tokens = config_json["eos_token_id"]
    return stop_tokens


def main(
    gui_host: str = "localhost",
    gui_port: int = 8001,
    oai_host: str = "https://brianfitzgerald--vllm-server-serve.modal.run/v1",
    oai_port: int = 8000,
    model: Optional[str] = None,
):
    def _get_model_id():
        logger.info(f"Getting model id, base url {client.base_url}")
        all_models = client.models.list()

        logger.info(f"Available models: {[x.id for x in all_models.data]}")
        first_model_id = all_models.data[0].id
        model_id = model or first_model_id
        logger.info(f"Selected model: {model_id}")
        return model_id

    def predict(message: str, history: list[tuple[str, str]]):
        nonlocal selected_model_id
        if selected_model_id is None:
            selected_model_id = _get_model_id()
        # Convert chat history to OpenAI format
        history_openai_format: Iterable[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a great ai assistant."}
        ]
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})  # type: ignore
            history_openai_format.append({"role": "assistant", "content": assistant})  # type: ignore
        history_openai_format.append({"role": "user", "content": message})  # type: ignore

        stop_tokens = _get_eos_tokens()

        # Create a chat completion request and send it to the API server
        logger.info(f"Sending message: {message} to model: {selected_model_id}")
        stream = client.chat.completions.create(
            messages=history_openai_format,
            model=selected_model_id,
            stream=True,
            extra_body={"stop_token_ids": stop_tokens, "skip_special_tokens": False},
        )

        # Read and return generated text from response stream
        partial_message = ""
        for chunk in stream:
            partial_message += chunk.choices[0].delta.content or ""
            yield partial_message

    oai_base_url = oai_host
    logger.info(f"Creating OpenAI client with base url: {oai_base_url}")
    client = OpenAI(base_url=oai_base_url, api_key="super-secret-key")
    selected_model_id = _get_model_id()

    # Create and launch a chat interface with Gradio
    gr.ChatInterface(
        predict,
        examples=[
            "Can you tell me a joke?",
            "Solve FizzBuzz in Python",
            "Write a story about a man and his cat, in a post-apocalyptic world.",
            "How many Rs are there in the word strawberry?",
        ],
    ).queue().launch(server_name=gui_host, server_port=gui_port)


if __name__ == "__main__":
    fire.Fire(main)
