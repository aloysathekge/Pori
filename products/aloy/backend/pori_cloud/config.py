from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./pori_cloud.db"

    # CORS
    cors_origins: str = (
        "http://localhost:3000,http://localhost:5173,https://app.pori.aloysathekge.com"  # comma-separated
    )

    # Logging
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_rpm: int = 60
    max_concurrent_runs: int = 5

    # Durable worker
    worker_poll_seconds: float = 1.0
    worker_lease_seconds: int = 900
    # How often the worker loop checks for due cron jobs
    cron_tick_seconds: float = 20.0

    # Messaging gateway (pori-cloud-gateway). Telegram is enabled by setting
    # the bot token; no token -> the adapter simply doesn't exist.
    telegram_bot_token: str = ""
    gateway_idle_sleep_seconds: float = 1.0
    gateway_error_backoff_seconds: float = 5.0
    gateway_pairing_ttl_seconds: int = 600

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def supabase_jwks_url(self) -> str:
        return f"{self.supabase_url}/auth/v1/.well-known/jwks.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
