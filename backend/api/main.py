"""FastAPI app: lifespan, route registration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes_rate import router
from backend.config import get_settings
from backend.db.migrations import run as run_migrations

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    run_migrations(settings.db_path)
    import backend.db.repository as repo
    import sqlite3
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    repo.set_connection(conn)
    logger.info("DB initialised at %s", settings.db_path)
    yield
    conn.close()


app = FastAPI(title="paper-scout", lifespan=lifespan)
app.include_router(router)
