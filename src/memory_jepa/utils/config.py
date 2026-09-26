from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted KEY=YAML_VALUE CLI overrides without silently creating typos."""
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        target: dict[str, Any] = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown config path: {dotted_key}")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown config key: {dotted_key}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return result

