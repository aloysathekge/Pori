"""Backend settings via pydantic-settings: one ``Settings`` class read from
the environment / ``.env`` (database URL, Supabase, CORS, rate limits, worker
and cron cadence, sandbox backend). Import the module-level ``settings``
singleton rather than instantiating ``Settings`` again.
"""

from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Bridge the dev ``.env`` into the process environment BEFORE anything reads it.
# pydantic-settings loads ``.env`` into ``Settings`` fields only — it never
# populates ``os.environ``. Kernel env-gated tools (web_search on
# SERPER_API_KEY / TAVILY_API_KEY, etc.) read ``os.getenv`` directly, so without
# this they can't see operator keys set in ``.env`` during local dev. In prod
# these are real env vars; ``load_dotenv`` never overrides an already-set one.
load_dotenv()


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

    # Conversation longevity. The model may have a much larger provider
    # context, but Event transcript latency must not grow without bound.
    conversation_history_window_tokens: int = Field(default=24_000, ge=512)
    conversation_hydration_max_chars: int = Field(default=192_000, ge=8_000)
    conversation_hydration_max_messages: int = Field(default=2_000, ge=50, le=10_000)
    event_history_search_max_candidates: int = Field(default=500, ge=50, le=2_000)

    # Durable worker
    worker_poll_seconds: float = 1.0
    worker_lease_seconds: int = 900
    # How often the worker loop checks for due cron jobs
    cron_tick_seconds: float = 20.0
    # Model-free live Surface health checks. Disabled until an operator has a
    # trusted browser inspection provider and accepts its sandbox cost.
    surface_reinspection_enabled: bool = False
    surface_reinspection_tick_seconds: float = Field(default=300.0, ge=30.0)
    surface_reinspection_interval_seconds: int = Field(default=86_400, ge=3_600)

    # Global Event-template catalog authority. Organization-level operator
    # permissions are necessary but not sufficient because catalog publication
    # affects every tenant. Comma-separated authenticated subject ids; empty is
    # fail-closed and disables catalog authoring endpoints.
    event_template_catalog_operator_subjects: str = ""

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

    # Generated Surface compilation is a separate authority boundary from
    # agent shell execution. Production defaults to an isolated provider.
    # `local_dev` runs only Aloy's fixed compiler command on the host and is
    # intended solely for a developer workstation without remote sandbox access.
    surface_build_backend: Literal["isolated", "local_dev"] = "isolated"

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

    # Event setup context ingestion. These jobs are model-independent and use
    # the same durable worker process as Runs, with their own short leases.
    context_ingestion_lease_seconds: int = 90
    context_ingestion_link_timeout_seconds: float = 10.0
    context_ingestion_max_link_bytes: int = 2 * 1024 * 1024
    context_ingestion_max_text_chars: int = 200_000

    # Aloy-owned specialist model assignments. The YAML contains provider/model
    # ids and qualification evidence only; credentials remain environment vars.
    aloy_model_roles_path: str = "aloy.models.yaml"

    # Every custom Event receives a published baseline Surface at creation via
    # the model-free materialization pipeline (docs/aloy-baseline-surface-spec.md).
    # The flag exists for rollout control and test isolation, not as a
    # per-user preference.
    surface_baseline_enabled: bool = True

    # Aggregate token ceiling for one Surface Builder Run. The default is the
    # V1 product cap; deployments whose Builder provider has no prompt caching
    # may need a higher ceiling, because the multi-turn workspace loop re-sends
    # its full growing context on every turn and the budget ledger currently
    # counts those repeated input tokens at full price.
    surface_builder_max_total_tokens: int = 200_000

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

    @property
    def event_template_catalog_operator_subjects_set(self) -> frozenset[str]:
        return frozenset(
            subject.strip()
            for subject in self.event_template_catalog_operator_subjects.split(",")
            if subject.strip()
        )


settings = Settings()
