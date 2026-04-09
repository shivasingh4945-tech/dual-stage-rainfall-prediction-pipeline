from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def resolve_project_path(path):
    """Resolve a project-relative path to an absolute path."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_config_path(path="configs/config.yaml"):
    """Locate the config file regardless of whether the caller runs from root or notebooks/."""
    path = Path(path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {path}")

    candidates = [
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (PROJECT_ROOT / "configs" / path.name).resolve(),
        DEFAULT_CONFIG_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config file not found. Checked: {candidates}")


def load_config(path="configs/config.yaml"):
    """
    Load config and normalize all configured paths to absolute project-root paths.
    """
    config_path = resolve_config_path(path)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {
        key: str(resolve_project_path(value))
        for key, value in cfg["paths"].items()
    }
    cfg["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
    }
    return cfg
