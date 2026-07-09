"""index the worker claim / admission queries on runs

Revision ID: q3a4b5c6d7e8
Revises: p2f3a4b5c6d7
Create Date: 2026-07-09
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "q3a4b5c6d7e8"
down_revision: Union[str, Sequence[str], None] = "p2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Worker claim query: WHERE cancel_requested/status/lease_expires_at
    # ORDER BY created_at (worker.py claim_next_run), and the send_message
    # admission count on status. Without this, every poll tick full-scans runs.
    op.create_index(
        "ix_runs_claim",
        "runs",
        ["status", "cancel_requested", "lease_expires_at", "created_at"],
    )
    op.create_index("ix_runs_status", "runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_runs_claim", table_name="runs")
