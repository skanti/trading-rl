"""Create or update the dashboard's Firebase Auth user from ``dashboard/config.yaml``.

The dashboard login form asks for a bare username; Firebase Auth needs an email
address. The two are bridged by ``auth.email_domain``, so the credentials in the config
file and the credentials Firebase will accept can never drift -- rerun this after
changing either the username or the password.

    ../ml/.venv/bin/python scripts/provision_auth_user.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

# The config contract lives with the other Python readers of config.yaml, so this script
# borrows it rather than growing a second copy that could drift from them.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))

from omegaconf import OmegaConf  # noqa: E402

from baseline.dashboard_config import (  # noqa: E402
    auth_email,
    load_dashboard_config,
    service_account,
)

LOGGER = logging.getLogger("dashboard-auth")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision the dashboard's Firebase Auth user.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="path to dashboard/config.yaml",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    config = load_dashboard_config(args.config)
    email = auth_email(config)
    password = str(OmegaConf.select(config, "auth.password") or "")
    if len(password) < 6:
        parser.error("auth.password must be at least 6 characters; Firebase rejects shorter ones")

    import firebase_admin
    from firebase_admin import auth, credentials

    project_id = str(OmegaConf.select(config, "firebase.project_id"))
    credential = service_account(config)
    if isinstance(credential, dict):
        certificate = credentials.Certificate(credential)
    elif credential is not None and credential.exists():
        certificate = credentials.Certificate(str(credential))
    else:
        parser.error(
            f"no Firebase service-account key found at {credential}; download one from "
            "Firebase console -> Project settings -> Service accounts"
        )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(certificate, {"projectId": project_id})

    try:
        user = auth.get_user_by_email(email)
    except auth.UserNotFoundError:
        user = auth.create_user(email=email, password=password, email_verified=True)
        LOGGER.info("created dashboard user %s (uid %s)", email, user.uid)
        return 0

    auth.update_user(user.uid, password=password, email_verified=True)
    LOGGER.info("updated password for existing dashboard user %s (uid %s)", email, user.uid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
