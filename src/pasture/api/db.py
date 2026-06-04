"""
SQLAlchemy persistence layer for inference results.

Failures are logged but never propagate to the caller — DB unavailability
must not take down the inference endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://pasture:pasture@postgres:5432/pasture",
)


class Base(DeclarativeBase):
    pass


class InferenceRecord(Base):
    __tablename__ = "inferences"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    bbox          = Column(String)   # JSON array string
    date_from     = Column(String)
    date_to       = Column(String)
    health_score  = Column(Float)
    ndvi_mean     = Column(Float)
    ndvi_std      = Column(Float)
    class_healthy = Column(Float)
    class_stressed= Column(Float)
    class_bare    = Column(Float)
    class_water   = Column(Float)
    inference_ms  = Column(Integer)


def _make_engine():
    try:
        engine = create_engine(_DATABASE_URL, pool_pre_ping=True, pool_size=2)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        log.warning("DB not available, persistence disabled: %s", e)
        return None


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def save_inference(result: dict, bbox: list, date_from: str, date_to: str) -> None:
    """Write one inference result row to PostgreSQL. Silently ignores errors."""
    engine = get_engine()
    if engine is None:
        return
    try:
        with Session(engine) as session:
            record = InferenceRecord(
                bbox          = json.dumps(bbox),
                date_from     = date_from,
                date_to       = date_to,
                health_score  = result["health_score"],
                ndvi_mean     = result["ndvi_stats"]["mean"],
                ndvi_std      = result["ndvi_stats"]["std"],
                class_healthy = result["class_pct"].get("healthy_pasture", 0.0),
                class_stressed= result["class_pct"].get("stressed", 0.0),
                class_bare    = result["class_pct"].get("bare", 0.0),
                class_water   = result["class_pct"].get("water", 0.0),
                inference_ms  = result["inference_ms"],
            )
            session.add(record)
            session.commit()
    except Exception as e:
        log.warning("Failed to persist inference: %s", e)
