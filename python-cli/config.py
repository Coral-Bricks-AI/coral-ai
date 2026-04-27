"""Persistent CLI config at `~/.coralbricks/config.json` (mode 0600).

Stores the current API key, server URL, and cached user identity. Reads
are synchronous and cheap — the file is small and loaded on every
command. Writes are atomic via rename.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_dir

DEFAULT_SERVER_URL = "https://backend.coralbricks.ai"
ENV_SERVER_URL = "CORALBRICKS_SERVER_URL"
ENV_API_KEY = "CORALBRICKS_API_KEY"


@dataclass
class Config:
    api_key: str | None = None
    server_url: str = DEFAULT_SERVER_URL
    user_id: int | None = None
    email: str | None = None
    extra: dict = field(default_factory=dict)

    def effective_server_url(self) -> str:
        return os.environ.get(ENV_SERVER_URL) or self.server_url or DEFAULT_SERVER_URL

    def effective_api_key(self) -> str | None:
        return os.environ.get(ENV_API_KEY) or self.api_key


def config_dir() -> Path:
    return Path(user_config_dir("coralbricks", appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.json"


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return Config()
    return Config(
        api_key=data.get("api_key"),
        server_url=data.get("server_url", DEFAULT_SERVER_URL),
        user_id=data.get("user_id"),
        email=data.get("email"),
        extra=data.get("extra", {}),
    )


def save(cfg: Config) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def clear() -> bool:
    path = config_path()
    if path.exists():
        path.unlink()
        return True
    return False
