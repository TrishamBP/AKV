"""Select a model's token counter behind the common ChatTokenCounter interface."""

import json
from importlib import import_module
from pathlib import Path

from ..types import ChatTokenCounter

MODEL_COUNTERS = {
    "gpt_oss": ("gpt_oss", "HarmonyTokenCounter"),
    "minimax_m2": ("minimax", "MiniMaxTokenCounter"),
}


def load_counter(model_path: str | Path) -> ChatTokenCounter:
    """Select a counter by model type; unmapped models use the Jinja counter."""
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    module_name, class_name = MODEL_COUNTERS.get(config.get("model_type"), ("jinja", "JinjaTokenCounter"))
    counter_class = getattr(import_module(f".{module_name}", __name__), class_name)
    return counter_class(model_path)
