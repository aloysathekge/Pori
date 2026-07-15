"""give every Event one durable user-facing conversation

Revision ID: v8f9a0b1c2d3
Revises: u7e8f9a0b1c2
Create Date: 2026-07-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "v8f9a0b1c2d3"
down_revision: Union[str, Sequence[str], None] = "u7e8f9a0b1c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.add_column(
            sa.Column("primary_conversation_id", sa.String(), nullable=True)
        )
        batch.create_index(
            "ix_events_primary_conversation_id", ["primary_conversation_id"]
        )

    connection = op.get_bind()
    events = sa.table(
        "events",
        sa.column("id", sa.String()),
        sa.column("primary_conversation_id", sa.String()),
    )
    conversations = sa.table(
        "conversations",
        sa.column("id", sa.String()),
        sa.column("event_id", sa.String()),
        sa.column("updated_at", sa.DateTime(timezone=True)),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    event_ids = connection.execute(sa.select(events.c.id)).scalars().all()
    for event_id in event_ids:
        conversation_id = connection.execute(
            sa.select(conversations.c.id)
            .where(conversations.c.event_id == event_id)
            .order_by(
                conversations.c.updated_at.desc(), conversations.c.created_at.desc()
            )
            .limit(1)
        ).scalar_one_or_none()
        if conversation_id:
            connection.execute(
                events.update()
                .where(events.c.id == event_id)
                .values(primary_conversation_id=conversation_id)
            )


def downgrade() -> None:
    with op.batch_alter_table("events") as batch:
        batch.drop_index("ix_events_primary_conversation_id")
        batch.drop_column("primary_conversation_id")
