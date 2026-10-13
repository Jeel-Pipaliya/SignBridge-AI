"""
SignBridge AI - Configuration Utility
Loads YAML configuration and resolves filesystem paths.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = (PROJECT_ROOT / "ai" / "configs") if (PROJECT_ROOT / "ai" / "configs").exists() else (PROJECT_ROOT / "configs")
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "training.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads configuration YAML file into a dictionary.
    """
    path = Path(config_path)
    if not path.is_absolute():
        resolved = PROJECT_ROOT / path
        if not resolved.exists() and (PROJECT_ROOT / "ai" / path).exists():
            resolved = PROJECT_ROOT / "ai" / path
        path = resolved

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def get_resolved_path(rel_or_abs_path: str) -> Path:
    """
    Resolves relative path against project root or ai/ directory.
    """
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p
    target = (PROJECT_ROOT / p).resolve()
    if not target.exists() and (PROJECT_ROOT / "ai" / p).resolve().exists():
        return (PROJECT_ROOT / "ai" / p).resolve()
    return target
