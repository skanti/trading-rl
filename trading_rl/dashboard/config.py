"""Load the dashboard's local YAML configuration.

It holds the Firebase project, service account, dashboard login and public URL.

Alpaca credentials are deliberately not here. The dashboard daemon reads
``ALPACA_KEY``/``ALPACA_SECRET`` from its environment.

Firebase credentials remain external to the installed package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_REPOSITORY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "config.yaml"


def _default_config_path() -> Path:
    if _REPOSITORY_CONFIG_PATH.is_file():
        return _REPOSITORY_CONFIG_PATH
    candidates = (Path.cwd() / "dashboard" / "config.yaml", Path.cwd() / "config.yaml")
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


DEFAULT_CONFIG_PATH = _default_config_path()
REQUIRED_KEYS = (
    "dashboard.url",
    "firebase.project_id",
    "firebase.collection",
    "firebase.document",
)


class DashboardConfigError(RuntimeError):
    """The dashboard configuration is missing or incomplete."""


def config_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the configuration path, honouring an explicit override then the env."""
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("DASHBOARD_CONFIG")
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_CONFIG_PATH


def load_dashboard_config(explicit: str | os.PathLike[str] | None = None) -> DictConfig:
    """Load and validate the dashboard configuration."""
    path = config_path(explicit)
    if not path.exists():
        raise DashboardConfigError(f"dashboard configuration not found at {path}")
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise DashboardConfigError(f"{path} must contain a YAML mapping at the top level")

    missing = [key for key in REQUIRED_KEYS if OmegaConf.select(config, key) is None]
    if missing:
        raise DashboardConfigError(
            f"{path} is missing required key(s): {', '.join(missing)}"
        )

    return config


def service_account(config: DictConfig) -> Path | dict[str, Any] | None:
    """The Firebase Admin SDK credential.

    Accepts either a path to the downloaded key file or the key JSON pasted inline as a
    mapping, so a deployment can keep everything in ``config.yaml``.
    """
    raw = OmegaConf.select(config, "firebase.service_account")
    if raw is None:
        return None
    if isinstance(raw, DictConfig):
        return OmegaConf.to_container(raw, resolve=True)  # type: ignore[return-value]
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    return Path(text).expanduser() if text else None


def service_account_path(config: DictConfig) -> Path | None:
    """Path form of the service-account credential, when it is configured as a path."""
    credential = service_account(config)
    return credential if isinstance(credential, Path) else None


def auth_email(config: DictConfig) -> str:
    """The Firebase Auth email derived from the dashboard username.

    The login form asks for a bare username; Firebase Auth needs an email address, so
    the two are bridged by a configured domain rather than a second stored credential.
    """
    username = OmegaConf.select(config, "auth.username")
    domain = OmegaConf.select(config, "auth.email_domain")
    if not username or not domain:
        raise DashboardConfigError("auth.username and auth.email_domain must both be set")
    return f"{username}@{domain}"


def as_plain(config: DictConfig) -> dict[str, Any]:
    """Resolved plain-dict view, for logging and tests."""
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
