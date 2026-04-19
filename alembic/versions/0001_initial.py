"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "own_accounts",
        sa.Column("slug", sa.String(64), primary_key=True),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("institution", sa.String(64), nullable=False),
        sa.Column("account_number_tail", sa.String(32)),
        sa.Column("aliases", sa.JSON),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "labeled_transactions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.String(128), nullable=False, unique=True),
        sa.Column("account_slug", sa.String(64), sa.ForeignKey("own_accounts.slug"), nullable=False),
        sa.Column("tx_date", sa.Date, nullable=False),
        sa.Column("amount", sa.Numeric(15, 2), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="COP"),
        sa.Column("transaction_type", sa.String(8), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("normalized_description", sa.Text, nullable=False),
        sa.Column("category_slug", sa.String(96), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="user"),
        sa.Column("embedding", Vector(384)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_labeled_transactions_category", "labeled_transactions", ["category_slug"])
    op.create_index("ix_labeled_transactions_account", "labeled_transactions", ["account_slug"])
    # IVFFlat over cosine. lists=100 is a reasonable default for <100k rows.
    op.execute(
        "CREATE INDEX ix_labeled_transactions_embedding "
        "ON labeled_transactions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("account_slug", sa.String(64), nullable=False),
        sa.Column("normalized_description", sa.Text, nullable=False),
        sa.Column("predicted_slug", sa.String(96), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("source_tier", sa.String(16), nullable=False),
        sa.Column("latency_ms", sa.Numeric(10, 2), nullable=False),
        sa.Column("trace", sa.JSON),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_external_id", "predictions", ["external_id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])

    op.create_table(
        "corrections",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("old_slug", sa.String(96)),
        sa.Column("new_slug", sa.String(96), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_corrections_external_id", "corrections", ["external_id"])

    op.create_table(
        "merchants",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("canonical_name", sa.String(128), nullable=False, unique=True),
        sa.Column("aliases", sa.JSON),
        sa.Column("mcc_hint", sa.String(8)),
        sa.Column("default_category_slug", sa.String(96)),
        sa.Column("source", sa.String(16), nullable=False, server_default="seed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("merchants")
    op.drop_index("ix_corrections_external_id", table_name="corrections")
    op.drop_table("corrections")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_external_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_labeled_transactions_embedding", table_name="labeled_transactions")
    op.drop_index("ix_labeled_transactions_account", table_name="labeled_transactions")
    op.drop_index("ix_labeled_transactions_category", table_name="labeled_transactions")
    op.drop_table("labeled_transactions")
    op.drop_table("own_accounts")
