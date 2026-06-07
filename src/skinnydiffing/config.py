"""YAML config loading and deep-merge expansion."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """
    Combine two dictionaries recursively, prioritising values from the override
    dictionary.

    Nested dictionaries are traversed and merged at every level. If a key exists in both
    dictionaries but the values are not dictionaries, the value from the override
    dictionary completely replaces the value from the base dictionary. Lists are replaced
    entirely, not concatenated. The original dictionaries are not modified.

    Args:
        base: The foundation dictionary containing default values.
        override: The dictionary containing specific values that replace the defaults.

    Returns:
        dict: A new dictionary containing the merged result.
    """
    result = copy.deepcopy(base)
    _merge_in_place(result, override)
    return result


def _is_loader_call(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 1


def _merge_in_place(base: dict, override: dict) -> None:
    for k, v in override.items():
        # Prevent merging different loaders (e.g. merging 'spark' with 'extract')
        if (
            k in {"source", "target"}
            and k in base
            and _is_loader_call(base[k])
            and _is_loader_call(v)
            and next(iter(base[k])) != next(iter(v))
        ):
            base[k] = copy.deepcopy(v)

        elif k in base and isinstance(v, dict) and isinstance(base[k], dict):
            _merge_in_place(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def expand_comparisons(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand configuration with global defaults and distinct jobs into standalone job
    configurations.

    Top-level keys and keys nested under a `defaults` block are deep-merged into every
    dictionary listed in the `comparisons` block. Overrides in the individual comparison
    blocks take precedence over the defaults.

    For example, an input like this::

        defaults:
            keys: ["id"]
            batch_size: 100
        comparisons:
            - source: "source_data.csv"
              target: "target_data.csv"
            - source: "source_other.csv"
              target: "target_other.csv"
              batch_size: 50

    Produces the following output::

        [
            {
                "keys": ["id"],
                "batch_size": 100,
                "source": "source_data.csv",
                "target": "target_data.csv",
            },
            {
                "keys": ["id"],
                "batch_size": 50,
                "source": "source_other.csv",
                "target": "target_other.csv",
            },
        ]

    Args:
        raw (dict): The raw dictionary parsed from the YAML configuration file.

    Returns:
        list[dict]: A list of fully expanded configuration dictionaries, one for each
            item in the `comparisons` list.

    Raises:
        TypeError: If the 'defaults' key is not a dictionary, or the 'comparisons' key
            is not a list.
    """
    top_level_defaults = {
        k: v for k, v in raw.items() if k not in {"defaults", "comparisons"}
    }
    named_defaults = raw.get("defaults", {})
    if named_defaults is None:
        named_defaults = {}
    if not isinstance(named_defaults, dict):
        raise TypeError(
            f"'defaults' must be a mapping, got {type(named_defaults).__name__}"
        )

    base = deep_merge(named_defaults, top_level_defaults)
    items = raw.get("comparisons", [{}])
    if not isinstance(items, list):
        raise TypeError(f"'comparisons' must be a list, got {type(items).__name__}")
    return [deep_merge(base, item) for item in items]


def load_config(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a YAML configuration file from disk and expand it into a list of standalone
    job dictionaries.

    The file is parsed into a Python dictionary and immediately passed through the
    expansion logic to merge any global or block-level defaults into the individual
    comparison jobs.

    Args:
        path: The file path to the YAML configuration file.

    Returns:
        list[dict]: A list of fully expanded configuration dictionaries, one for each
            comparison job defined in the file.

    Raises:
        ValueError: If the file is completely empty.
        TypeError: If the parsed YAML does not evaluate to a top-level dictionary.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file {path!s} is empty")
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__}")
    return expand_comparisons(raw)
