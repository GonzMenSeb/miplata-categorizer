"""FastAPI entry."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from .api import router
from .config import get_settings
from .logging_setup import configure_logging
from .taxonomy import load_taxonomy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> "AsyncIterator[None]":
    settings = get_settings()
    configure_logging(settings.log_level)

    # Load taxonomy once at startup; it's read-only afterward.
    app.state.taxonomy = load_taxonomy(settings.taxonomy_path)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="miplata-categorizer",
        version="0.1.0",
        description="Self-hosted Colombian-Spanish tx categorizer (cascade).",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
