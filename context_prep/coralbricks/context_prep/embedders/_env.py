"""Shared dotenv discovery for embedders.

Embedders historically read API keys from ``www/ml/embed/.env``. Now that
the embedder code lives under ``coralbricks/context_prep/embedders``, we
look in a few candidate locations so existing user setups keep working.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_CANDIDATE_REL_PATHS: tuple[str, ...] = (
    # New SDK location ($CWD/.env or platform/.env)
    ".env",
    # Legacy embedder home
    "../../../../embed/.env",  # coralbricks/context_prep/embedders -> www/ml/embed
    "../../embed/.env",         # repo-root anchored fallback
)


def _candidate_paths() -> Iterable[Path]:
    here = Path(__file__).resolve().parent
    for rel in _CANDIDATE_REL_PATHS:
        yield (here / rel).resolve()
    # Also try $CORAL_EMBED_DOTENV if set explicitly
    explicit = os.environ.get("CORAL_EMBED_DOTENV")
    if explicit:
        yield Path(explicit).expanduser().resolve()


def load_embed_dotenv() -> None:
    """Best-effort load of ``.env`` for embedder API keys.

    Silently no-ops if ``python-dotenv`` is not installed or no candidate
    file exists.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    for candidate in _candidate_paths():
        try:
            if candidate.exists():
                load_dotenv(candidate)
                return
        except OSError:
            continue
