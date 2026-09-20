"""
SignBridge AI - Configuration Utility
Loads YAML configuration and resolves filesystem paths.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "training.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads configuration YAML file into a dictionary.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


def get_resolved_path(rel_or_abs_path: str) -> Path:
    """
    Resolves relative path against project root.
    """
    p = Path(rel_or_abs_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
