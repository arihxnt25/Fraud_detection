"""Persistence for scored transactions and analyst cases.

Uses MongoDB when ``MONGO_URI`` is set (see docker-compose.yml); otherwise
falls back to an in-memory store so the API runs with zero setup.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class MemoryStore:
    backend = "memory"

    def __init__(self):
        self._scores: list[dict] = []
        self._cases: dict[str, dict] = {}

    def save_score(self, doc: dict) -> None:
        self._scores.append({**doc, "scored_at": _now()})

    def recent_scores(self, limit: int = 100, min_tier: str | None = None) -> list[dict]:
        rows = self._scores[::-1]
        if min_tier:
            rows = [r for r in rows if _tier_rank(r["risk_tier"]) >= _tier_rank(min_tier)]
        return rows[:limit]

    def upsert_case(self, case_id: str, doc: dict) -> dict:
        case = self._cases.setdefault(case_id, {"case_id": case_id, "created_at": _now()})
        case.update(doc, updated_at=_now())
        return case

    def list_cases(self) -> list[dict]:
        return list(self._cases.values())


class MongoStore(MemoryStore):
    backend = "mongodb"

    def __init__(self, uri: str, db: str = "fraudnet"):
        from pymongo import MongoClient
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.client.admin.command("ping")
        self.db = self.client[db]
        self.db.scores.create_index("transaction_id")
        self.db.scores.create_index([("risk_score", -1)])

    def save_score(self, doc: dict) -> None:
        self.db.scores.insert_one({**doc, "scored_at": _now()})

    def recent_scores(self, limit: int = 100, min_tier: str | None = None) -> list[dict]:
        q = {}
        if min_tier:
            q = {"risk_tier": {"$in": [t for t in TIERS if _tier_rank(t) >= _tier_rank(min_tier)]}}
        return list(self.db.scores.find(q, {"_id": 0}).sort("scored_at", -1).limit(limit))

    def upsert_case(self, case_id: str, doc: dict) -> dict:
        self.db.cases.update_one({"case_id": case_id},
                                 {"$set": {**doc, "updated_at": _now()},
                                  "$setOnInsert": {"created_at": _now()}}, upsert=True)
        return self.db.cases.find_one({"case_id": case_id}, {"_id": 0})

    def list_cases(self) -> list[dict]:
        return list(self.db.cases.find({}, {"_id": 0}))


TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _tier_rank(t: str) -> int:
    return TIERS.index(t.upper()) if t and t.upper() in TIERS else 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_store():
    uri = os.getenv("MONGO_URI")
    if uri:
        try:
            return MongoStore(uri)
        except Exception as e:  # pragma: no cover - depends on infra
            log.warning("MongoDB unavailable (%s); using in-memory store", e)
    return MemoryStore()
