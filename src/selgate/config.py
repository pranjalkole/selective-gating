"""Tiny YAML config loader with attribute access and dot-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml


class Config(dict):
    """A dict whose string keys are also readable/writable as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        out = cls()
        for k, v in (d or {}).items():
            out[k] = cls.from_dict(v) if isinstance(v, dict) else v
        return out

    def to_dict(self) -> Dict[str, Any]:
        def conv(v: Any) -> Any:
            if isinstance(v, Config):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v

        return conv(self)

    def get_path(self, dot_key: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dot_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, dot_key: str, value: Any) -> None:
        parts = dot_key.split(".")
        cur: Any = self
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = Config()
            cur = cur[p]
        cur[parts[-1]] = value

    def __deepcopy__(self, memo: Dict) -> "Config":
        return Config.from_dict(copy.deepcopy(self.to_dict()))


def _parse_scalar(s: str) -> Any:
    """Parse a CLI override value with YAML semantics ('1e-4' -> float, 'true' -> bool)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> Config:
    """Load a YAML config; apply ``key.sub=value`` overrides (repeatable)."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config.from_dict(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key=value, got: {ov!r}")
        key, val = ov.split("=", 1)
        cfg.set_path(key.strip(), _parse_scalar(val.strip()))
    return cfg
