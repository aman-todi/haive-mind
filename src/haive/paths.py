"""Filesystem locations for config and data.

`HAIVE_HOME` overrides both (used by tests and for portable installs):
config goes to `$HAIVE_HOME/config`, data to `$HAIVE_HOME/data`.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP = "haive"


def config_dir() -> Path:
    home = os.environ.get("HAIVE_HOME")
    if home:
        return Path(home) / "config"
    return Path(user_config_dir(APP, appauthor=False))


def data_dir() -> Path:
    home = os.environ.get("HAIVE_HOME")
    if home:
        return Path(home) / "data"
    return Path(user_data_dir(APP, appauthor=False))


def config_file() -> Path:
    return config_dir() / "config.toml"


def env_file() -> Path:
    return config_dir() / ".env"


def db_file() -> Path:
    return data_dir() / "haive.db"


def checkpoints_file() -> Path:
    return data_dir() / "checkpoints.db"


def clones_dir() -> Path:
    return data_dir() / "clones"


def chroma_dir() -> Path:
    return data_dir() / "chroma"


def ensure_dirs() -> None:
    for d in (config_dir(), data_dir(), clones_dir()):
        d.mkdir(parents=True, exist_ok=True)
