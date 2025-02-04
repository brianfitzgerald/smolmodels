import os
import sys
import fire
import uvicorn

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    ServerSettings,
    ModelSettings,
    ConfigFileSettings,
)


def main(model: str, config_file: str):
    server_settings: ServerSettings | None = None
    model_settings: list[ModelSettings] = []
    try:
        # Load server settings from config_file if provided
        config_file = os.environ.get("CONFIG_FILE", config_file)
        if config_file:
            if not os.path.exists(config_file):
                raise ValueError(f"Config file {config_file} not found!")
            with open(config_file, "rb") as f:
                # Check if yaml file
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    import yaml
                    import json

                    config_file_settings = ConfigFileSettings.model_validate_json(
                        json.dumps(yaml.safe_load(f))
                    )
                else:
                    config_file_settings = ConfigFileSettings.model_validate_json(
                        f.read()
                    )
                server_settings = ServerSettings.model_validate(config_file_settings)
                model_settings = config_file_settings.models
        else:
            server_settings = ServerSettings()
            model_settings = [ModelSettings(model=model)]
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    assert server_settings is not None
    assert model_settings is not None
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", server_settings.host),
        port=int(os.getenv("PORT", server_settings.port)),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
    )


if __name__ == "__main__":
    fire.Fire(main)
