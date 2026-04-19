"""add miplata_account_id to own_accounts

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-18

miplata passes its Prisma Account UUID as `account_slug` when calling
/v1/categorize. The categorizer keys own_accounts on friendly slugs, so
without a translation column the paired-tx internal-transfer tier can't
resolve the incoming UUID to the correct own_account row.

This migration adds a nullable `miplata_account_id` UUID column plus a
unique index (one miplata account maps to at most one own_account).
Population happens out-of-band via `scripts/sync_miplata_accounts.py`.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "own_accounts",
        sa.Column("miplata_account_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_own_accounts_miplata_account_id",
        "own_accounts",
        ["miplata_account_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_own_accounts_miplata_account_id", table_name="own_accounts")
    op.drop_column("own_accounts", "miplata_account_id")
