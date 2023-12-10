import json
import os

with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    _config = json.load(f)

def get_config(model_name):
    model = _config["models"][model_name]
    chat_format = model["chat_format"]
    base_path = "base_path" in _config and _config["base_path"] or ""
    commons = _config["common"] if "common" in _config else {}
    return {
        "model_path": base_path + model["path"],
        "n_threads": "n_threads" in commons and commons["n_threads"] or 16,
        "n_gpu_layers": "n_gpu_layers" in commons and commons["n_gpu_layers"] or 1,
        "silence_warnings": "silence_warnings" in commons and commons["silence_warnings"] or True,
        "n_ctx": model["n_ctx"],
        "n_batch": "n_batch" in model and model["n_batch"] or 512,
        "chat_format": chat_format, 
        "verbose": True,
    }
