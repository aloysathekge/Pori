"""Backend settings via pydantic-settings: one ``Settings`` class read from
the environment / ``.env`` (database URL, Supabase, CORS, rate limits, worker
and cron cadence, sandbox backend). Import the module-level ``settings``
singleton rather than instantiating ``Settings`` again.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./aloy_backend.db"

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

    # Execution backend for agent shell/code: 'local' (runs on the worker
    # host) or 'e2b' (isolated cloud microVM per session — needs E2B_API_KEY
    # and the pori[sandbox-e2b] extra). Aloy-managed: users never configure
    # this; it's an operator setting reflected read-only in the app.
    sandbox_backend: str = "local"
    sandbox_enabled: bool = False
    # Filesystem jail for agent file tools + the local sandbox: every
    # conversation gets {sandbox_base_dir}/threads/{conversation_id}/user-data/
    # {workspace,uploads,outputs}. Always plumbed (even with the shell sandbox
    # disabled) so file tools never write to the host process cwd.
    sandbox_base_dir: str = ".aloy_sandbox"

    # Object storage — durable blobs (agent artifacts now; uploads in Phase 2).
    # 'local' = disk under storage_dir (dev default); 's3' arrives in Phase 3.
    storage_backend: str = "local"
    storage_dir: str = ".aloy_storage"
    storage_max_artifact_mb: int = 25  # per artifact file
    storage_max_run_artifact_mb: int = 100  # total per run
    storage_max_file_mb: int = 100  # per user upload
    storage_org_quota_mb: int = 2048  # total stored bytes per organization
    # S3-compatible backend (STORAGE_BACKEND=s3): AWS S3, Cloudflare R2,
    # MinIO, or Supabase Storage via its S3 protocol (Project Settings →
    # Storage → S3 access keys). Needs boto3 (optional dependency).
    storage_s3_endpoint: str = ""  # e.g. https://xxxx.supabase.co/storage/v1/s3
    storage_s3_bucket: str = "aloy-files"
    storage_s3_access_key: str = ""
    storage_s3_secret_key: str = ""
    storage_s3_region: str = "us-east-1"
    storage_presign_expiry_seconds: int = 300

    # Messaging gateway (aloy-backend-gateway). Telegram is enabled by setting
    # the bot token; no token -> the adapter simply doesn't exist.
    telegram_bot_token: str = ""
    gateway_idle_sleep_seconds: float = 1.0
    gateway_error_backoff_seconds: float = 5.0
    gateway_pairing_ttl_seconds: int = 600

    # Account connections (native OAuth — Gmail etc.). Tokens are encrypted at
    # rest with CONNECTIONS_ENC_KEY (a Fernet key: `python -c "from
    # cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
    # A provider is only offered when its client id/secret are set.
    connections_enc_key: str = ""
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    # Public base URLs for the OAuth redirect round-trip.
    backend_base_url: str = "http://localhost:8000"
    app_base_url: str = "http://localhost:5173"
    connection_flow_ttl_seconds: int = 600

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def supabase_jwks_url(self) -> str:
        return f"{self.supabase_url}/auth/v1/.well-known/jwks.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
