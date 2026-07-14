"""OAuth provider specs. Adding an app = add a spec here (+ its tools)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _credential(env_name: str) -> str:
    """Process env first, then Settings (which reads the .env file — values
    there never reach ``os.environ``, so env-only lookup missed them)."""
    value = os.getenv(env_name, "")
    if value:
        return value
    from ..config import settings

    return str(getattr(settings, env_name.lower(), "") or "")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    authorize_url: str
    token_url: str
    revoke_url: str
    userinfo_url: str  # returns the connected account's email
    scopes: List[str]
    client_id_env: str
    client_secret_env: str
    extra_authorize_params: Dict[str, str] = field(default_factory=dict)
    # A friendly label + description for the app's Connections screen.
    label: str = ""
    description: str = ""

    @property
    def client_id(self) -> str:
        return _credential(self.client_id_env)

    @property
    def client_secret(self) -> str:
        return _credential(self.client_secret_env)

    def is_configured(self) -> bool:
        """A provider is only offered when its OAuth app credentials are set."""
        return bool(self.client_id and self.client_secret)


GOOGLE = ProviderSpec(
    name="google",
    label="Google",
    description="Let the agent search, read, and send Gmail and manage your "
    "Google Calendar.",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    revoke_url="https://oauth2.googleapis.com/revoke",
    userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
    scopes=[
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        # gmail.compose is required for drafts.create (gmail.send only permits
        # *sending*, not saving a draft). It also covers sending, but we keep
        # gmail.send explicit so that capability is never lost if scopes change.
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ],
    client_id_env="GOOGLE_OAUTH_CLIENT_ID",
    client_secret_env="GOOGLE_OAUTH_CLIENT_SECRET",
    # offline + consent => Google returns a refresh_token we can persist.
    extra_authorize_params={"access_type": "offline", "prompt": "consent"},
)

PROVIDERS: Dict[str, ProviderSpec] = {GOOGLE.name: GOOGLE}


def get_provider(name: str) -> Optional[ProviderSpec]:
    return PROVIDERS.get((name or "").strip().lower())


def available_providers() -> List[ProviderSpec]:
    """Providers whose OAuth app credentials are configured on this deployment."""
    return [p for p in PROVIDERS.values() if p.is_configured()]
