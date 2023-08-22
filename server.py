import os
import warnings
import matplotlib
import json
import sys
import time
import yaml


from pathlib import Path
from threading import Lock

from app import shared
from app.utils import utils
from app.front import interface
from app.utils.logging import logger
from app.engine.LoRA import add_lora_to_model
from app.models import load_model
from app.settings import get_model_settings
from app.settings import update_model_parameters
import app.extensions as extensions_module


os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
matplotlib.use("Agg")  # This fixes LaTeX rendering on some systems


if __name__ == "__main__":
    # Loading custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path("settings.yaml").exists():
        settings_file = Path("settings.yaml")
    elif Path("settings.json").exists():
        settings_file = Path("settings.json")

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, "r", encoding="utf-8").read()
        new_settings = (
            json.loads(file_contents)
            if settings_file.suffix == "json"
            else yaml.safe_load(file_contents)
        )
        for item in new_settings:
            shared.settings[item] = new_settings[item]

    # Set default model settings based on settings file
    shared.model_config[".*"] = {
        "wbits": "None",
        "model_type": "None",
        "groupsize": "None",
        "pre_layer": 0,
        "mode": shared.settings["mode"],
        "skip_special_tokens": shared.settings["skip_special_tokens"],
        "custom_stopping_strings": shared.settings["custom_stopping_strings"],
        "truncation_length": shared.settings["truncation_length"],
    }

    shared.model_config.move_to_end(".*", last=False)  # Move to the beginning

    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
    if shared.is_chat():
        for extension in shared.settings["chat_default_extensions"]:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)
    else:
        for extension in shared.settings["default_extensions"]:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    # Default Models
    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error("No models are available! Please download at least one.")
            sys.exit(0)
        else:
            print("The following models are available:\n")
            for i, model in enumerate(available_models):
                print(f"{i+1}. {model}")

            print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != "None":
        model_settings = get_model_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(
            model_settings, initial=True
        )  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    # Forcing some events to be triggered on page load
    shared.persistent_interface_state.update(
        {
            "loader": shared.args.loader or "Transformers",
        }
    )

    if shared.is_chat():
        shared.persistent_interface_state.update(
            {
                "mode": shared.settings["mode"],
                "character_menu": shared.args.character or shared.settings["character"],
                "instruction_template": shared.settings["instruction_template"],
            }
        )

        if Path("cache/pfp_character.png").exists():
            Path("cache/pfp_character.png").unlink()

    shared.generation_lock = Lock()

    # Launch the web UI
    interface.create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio["interface"].close()
            time.sleep(0.5)
            interface.create_interface()
