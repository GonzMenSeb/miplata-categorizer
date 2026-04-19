"""Pytest bootstrap.

Set the env vars pydantic-settings requires so importing modules like
`categorizer.cascade` (which transitively touches get_settings) doesn't
fail at collection time. The values are placeholders; no test here
actually connects to Postgres or hits the LLM.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://stub:stub@localhost:5432/stub")
os.environ.setdefault("CATEGORIZER_API_TOKEN", "test-token")
