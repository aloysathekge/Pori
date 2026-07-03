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

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def supabase_jwks_url(self) -> str:
        return f"{self.supabase_url}/auth/v1/.well-known/jwks.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
