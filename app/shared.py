import argparse
import yaml

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional

from app.const import DEFAULT_SETTINGS
from app.utils.logging import logger
from app.cli import parser


generation_lock: Optional[Lock] = None
model = None
tokenizer = None
is_seq2seq = False
model_name = "None"
lora_names = []
model_dirty_from_training = False

# Chat variables
stop_everything = False
processing_message = "*Is typing...*"

# UI elements (buttons, sliders, HTML, etc)
gradio = {}

# For keeping the values of UI elements on page reload
persistent_interface_state = {}

input_params = []  # Generation input parameters
reload_inputs = []  # Parameters for reloading the chat interface

# For restarting the interface
need_restart = False

settings = DEFAULT_SETTINGS.copy()


args = parser.parse_args()
args_defaults = parser.parse_args([])

# Deprecation warnings
if args.autogptq:
    logger.warning(
        "--autogptq has been deprecated and will be removed soon. Use --loader autogptq instead."
    )
    args.loader = "autogptq"

if args.gptq_for_llama:
    logger.warning(
        "--gptq-for-llama has been deprecated and will be removed soon. Use --loader gptq-for-llama instead."
    )
    args.loader = "gptq-for-llama"

if args.flexgen:
    logger.warning(
        "--flexgen has been deprecated and will be removed soon. Use --loader flexgen instead."
    )
    args.loader = "FlexGen"

# Security warnings
if args.trust_remote_code:
    logger.warning("trust_remote_code is enabled. This is dangerous.")

if args.share:
    logger.warning(
        'The gradio "share link" feature uses a proprietary executable to create a reverse tunnel. Use it with care.'
    )

if args.multi_user:
    logger.warning("The multi-user mode is highly experimental. DO NOT EXPOSE IT TO THE INTERNET.")


def fix_loader_name(name):
    name = name.lower()
    if name in ["llamacpp", "llama.cpp", "llama-cpp", "llama cpp"]:
        return "llama.cpp"
    if name in [
        "llamacpp_hf",
        "llama.cpp_hf",
        "llama-cpp-hf",
        "llamacpp-hf",
        "llama.cpp-hf",
    ]:
        return "llamacpp_HF"
    elif name in ["transformers", "huggingface", "hf", "hugging_face", "hugging face"]:
        return "Transformers"
    elif name in ["autogptq", "auto-gptq", "auto_gptq", "auto gptq"]:
        return "AutoGPTQ"
    elif name in [
        "gptq-for-llama",
        "gptqforllama",
        "gptqllama",
        "gptq for llama",
        "gptq_for_llama",
    ]:
        return "GPTQ-for-LLaMa"
    elif name in ["exllama", "ex-llama", "ex_llama", "exlama"]:
        return "ExLlama"
    elif name in [
        "exllama-hf",
        "exllama_hf",
        "exllama hf",
        "ex-llama-hf",
        "ex_llama_hf",
    ]:
        return "ExLlama_HF"


if args.loader is not None:
    args.loader = fix_loader_name(args.loader)


def add_extension(name):
    if args.extensions is None:
        args.extensions = [name]
    elif "api" not in args.extensions:
        args.extensions.append(name)


# Activating the API extension
if args.api or args.public_api:
    add_extension("api")

# Activating the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension("multimodal")


def is_chat():
    return args.chat


def get_mode():
    if args.chat:
        return "chat"
    elif args.notebook:
        return "notebook"
    else:
        return "default"


# Loading model-specific settings
with Path(f"{args.model_dir}/config.yaml") as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, "r").read())
    else:
        model_config = {}

# Applying user-defined model settings
with Path(f"{args.model_dir}/config-user.yaml") as p:
    if p.exists():
        user_config = yaml.safe_load(open(p, "r").read())
        for k in user_config:
            if k in model_config:
                model_config[k].update(user_config[k])
            else:
                model_config[k] = user_config[k]

model_config = OrderedDict(model_config)
