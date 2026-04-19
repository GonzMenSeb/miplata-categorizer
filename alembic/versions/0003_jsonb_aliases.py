"""jsonb aliases

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-18

Converts merchants.aliases and own_accounts.aliases from JSON to JSONB so
the `?` key-exists operator (used by merchant.lookup) works. JSON has no
such operator; JSONB does.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "merchants",
        "aliases",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(),
        existing_nullable=True,
        postgresql_using="aliases::jsonb",
    )
    op.alter_column(
        "own_accounts",
        "aliases",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(),
        existing_nullable=True,
        postgresql_using="aliases::jsonb",
    )


def downgrade() -> None:
    op.alter_column(
        "own_accounts",
        "aliases",
        existing_type=postgresql.JSONB(),
        type_=sa.JSON(),
        existing_nullable=True,
        postgresql_using="aliases::json",
    )
    op.alter_column(
        "merchants",
        "aliases",
        existing_type=postgresql.JSONB(),
        type_=sa.JSON(),
        existing_nullable=True,
        postgresql_using="aliases::json",
    )
