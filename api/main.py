"""FastAPI service for real-time fraud scoring and fraud-network investigation.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudnet import __version__  # noqa: E402
from fraudnet.config import ARTIFACT_DIR  # noqa: E402
from fraudnet.db.store import get_store  # noqa: E402
from fraudnet.models.pipeline import FraudPipeline  # noqa: E402
from fraudnet.risk.scoring import RiskScorer  # noqa: E402

log = logging.getLogger("fraudnet.api")
STATE: dict = {}


def load_state(model_path: Path | None = None) -> None:
    path = Path(model_path or os.getenv("FRAUDNET_MODEL", ARTIFACT_DIR / "fraud_pipeline.joblib"))
    if not path.exists():
        raise RuntimeError(f"Model not found at {path}. Train first: "
                           "python scripts/run_pipeline.py --synthetic")
    pipe = FraudPipeline.load(path)
    STATE.update(pipeline=pipe, scorer=RiskScorer(pipe), store=get_store())


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "pipeline" not in STATE:
        load_state()
    yield


app = FastAPI(title="FraudNet API", version=__version__, lifespan=lifespan,
              description="Graph-augmented credit-card fraud detection with explainable risk scores.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ schemas
class Transaction(BaseModel):
    """Subset of IEEE-CIS fields; unknown extra fields are accepted and used."""
    model_config = ConfigDict(extra="allow", json_schema_extra={"example": {
        "TransactionID": 3663549, "TransactionDT": 15000000, "TransactionAmt": 249.0,
        "ProductCD": "C", "card1": 13926, "card2": 327.0, "card4": "visa", "card6": "debit",
        "addr1": 315.0, "D1": 0.0, "P_emaildomain": "anonymous.com",
        "DeviceType": "mobile", "DeviceInfo": "SM-G960U", "id_23": "IP_PROXY:ANONYMOUS"}})

    TransactionID: int | None = None
    TransactionDT: int = Field(..., description="seconds from the dataset reference time")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str | None = None
    card1: int
    card2: float | None = None
    card4: str | None = None
    card6: str | None = None
    addr1: float | None = None
    D1: float | None = 0.0
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    id_23: str | None = None


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=5000)
    explain: bool = True


class CaseUpdate(BaseModel):
    status: str = Field("open", pattern="^(open|investigating|confirmed_fraud|false_positive|closed)$")
    assignee: str | None = None
    notes: str | None = None
    transaction_ids: list[int] = []
    cluster_id: int | None = None


def _to_frame(txs: list[Transaction]) -> pd.DataFrame:
    df = pd.DataFrame([t.model_dump() for t in txs])
    df["TransactionID"] = df["TransactionID"].fillna(-1).astype(int)
    return df


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    pipe: FraudPipeline = STATE["pipeline"]
    return {"status": "ok", "version": __version__, "store": STATE["store"].backend,
            "model": {k: pipe.metrics.get(k) for k in ("roc_auc", "pr_auc", "n_features", "use_graph")},
            "graph": pipe.graph.summary() if pipe.graph else None}


@app.post("/score")
def score(tx: Transaction, explain: bool = True):
    res = STATE["scorer"].score(_to_frame([tx]), explain=explain)[0].to_dict()
    STATE["store"].save_score({k: v for k, v in res.items() if k != "top_features"})
    return res


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    results = [r.to_dict() for r in STATE["scorer"].score(_to_frame(req.transactions), req.explain)]
    for r in results:
        STATE["store"].save_score({k: v for k, v in r.items() if k != "top_features"})
    tiers = pd.Series([r["risk_tier"] for r in results]).value_counts().to_dict()
    return {"n": len(results), "flagged": sum(r["flagged"] for r in results),
            "tiers": tiers, "results": results}


@app.get("/alerts")
def alerts(limit: int = Query(100, le=1000), min_tier: str = "HIGH"):
    return STATE["store"].recent_scores(limit, min_tier)


@app.get("/clusters")
def clusters(min_cards: int = 3, min_fraud_rate: float = 0.3, top_k: int = Query(20, le=200)):
    g = STATE["pipeline"].graph
    if g is None:
        raise HTTPException(404, "model trained without graph features")
    return [c.to_dict() for c in g.fraud_clusters(min_cards, min_fraud_rate, top_k)]


@app.get("/graph/{node:path}")
def graph_neighbourhood(node: str, radius: int = Query(2, ge=1, le=3)):
    """Ego network around a node, e.g. ``device:SM-G960U`` or ``card:13926_315.0_150``."""
    g = STATE["pipeline"].graph
    try:
        return g.subgraph(node, radius=radius)
    except KeyError as err:
        raise HTTPException(404, f"node '{node}' not in graph") from err


@app.get("/model/importance")
def importance(top_k: int = 25):
    return STATE["pipeline"].feature_importance(top_k).to_dict(orient="records")


@app.get("/model/metrics")
def metrics():
    return STATE["pipeline"].metrics


@app.post("/cases")
def create_case(update: CaseUpdate):
    return STATE["store"].upsert_case(f"CASE-{uuid.uuid4().hex[:8].upper()}", update.model_dump())


@app.patch("/cases/{case_id}")
def update_case(case_id: str, update: CaseUpdate):
    return STATE["store"].upsert_case(case_id, update.model_dump(exclude_unset=True))


@app.get("/cases")
def list_cases():
    return STATE["store"].list_cases()
