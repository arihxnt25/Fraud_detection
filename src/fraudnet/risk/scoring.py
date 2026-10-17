"""Explainable risk scoring: blends ML probability, graph risk and expert rules.

Every score comes with human-readable reasons so an analyst can see *why* a
transaction was flagged (regulatory requirement in most card networks).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from fraudnet.config import RiskConfig
from fraudnet.features.engineering import add_card_uid
from fraudnet.models.pipeline import FraudPipeline

READABLE = {
    "g_component_fraud_rate": "linked to a cluster of known-fraud cards",
    "g_fraud_proximity": "close to known fraud in the entity network",
    "g_max_entity_fraud_rate": "shares a device/email/address with fraud cards",
    "g_device_fraud_rate": "device previously used by fraudulent cards",
    "g_email_payer_fraud_rate": "payer e-mail domain linked to fraud",
    "g_address_fraud_rate": "billing address linked to fraud",
    "amt_vs_uid_mean": "amount unusual for this card",
    "TransactionAmt": "transaction amount",
    "secs_since_prev_uid": "time since card's previous transaction",
    "uid_tx_count": "card transaction history length",
}


@dataclass
class RiskResult:
    transaction_id: int | None
    risk_score: float
    risk_tier: str
    model_probability: float
    graph_risk: float
    rule_risk: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    top_features: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def rule_risk(row: pd.Series) -> tuple[float, list[str]]:
    reasons, score = [], 0.0
    amt = float(row.get("TransactionAmt", 0) or 0)
    hour = (int(row.get("TransactionDT", 0)) // 3600) % 24
    if amt > 1000:
        score += 0.4; reasons.append(f"high amount (${amt:,.2f})")
    if hour < 6 and amt > 300:
        score += 0.3; reasons.append("large purchase between 00:00 and 06:00")
    if isinstance(row.get("id_23"), str) and "ANONYMOUS" in row["id_23"].upper():
        score += 0.3; reasons.append("anonymous IP proxy")
    if row.get("P_emaildomain") in {"anonymous.com", "protonmail.com", "mail.com"}:
        score += 0.2; reasons.append(f"high-risk e-mail domain ({row['P_emaildomain']})")
    return min(score, 1.0), reasons


class RiskScorer:
    def __init__(self, pipeline: FraudPipeline, cfg: RiskConfig | None = None):
        self.pipe = pipeline
        self.cfg = cfg or RiskConfig()

    def tier(self, score: float) -> str:
        for cut, name in self.cfg.tiers:
            if score >= cut:
                return name
        return "LOW"

    def score(self, df: pd.DataFrame, explain: bool = True) -> list[RiskResult]:
        df = add_card_uid(df.reset_index(drop=True))
        proba = self.pipe.predict_proba(df)
        if self.pipe.graph is not None:
            g = self.pipe.graph.transform(df)
            # rates are already Bayesian-smoothed; proximity is a softer signal
            graph_risk = np.clip(np.max(np.vstack([
                g["g_component_fraud_rate"].values,
                0.5 * g["g_fraud_proximity"].values,
                g.get("g_max_entity_fraud_rate", pd.Series(0, index=g.index)).values,
            ]), axis=0), 0, 1)
        else:
            g, graph_risk = None, np.zeros(len(df))
        expl = self.pipe.explain(df) if explain else [[] for _ in range(len(df))]

        results = []
        for i, row in df.iterrows():
            r_rule, reasons = rule_risk(row)
            s = (self.cfg.w_model * proba[i] + self.cfg.w_graph * graph_risk[i]
                 + self.cfg.w_rules * r_rule)
            # a very confident model or a known ring alone is enough to escalate
            s = float(max(s, proba[i] * 0.95, graph_risk[i] * 0.85 if graph_risk[i] > 0.6 else 0))
            if g is not None:
                if g.at[i, "g_component_fraud_rate"] >= 0.4:
                    reasons.insert(0, f"member of fraud cluster (size {int(g.at[i, 'g_component_size'])}, "
                                      f"~{g.at[i, 'g_component_fraud_rate']:.0%} fraud)")
                elif graph_risk[i] >= 0.4:
                    reasons.insert(0, READABLE["g_max_entity_fraud_rate"])
            for f in expl[i]:
                if f["contribution"] > 0.3 and f["feature"] in READABLE:
                    msg = READABLE[f["feature"]]
                    if msg not in reasons:
                        reasons.append(msg)
            results.append(RiskResult(
                transaction_id=int(row["TransactionID"])
                if "TransactionID" in row and pd.notna(row["TransactionID"])
                and row["TransactionID"] >= 0 else None,
                risk_score=round(s, 4), risk_tier=self.tier(s),
                model_probability=round(float(proba[i]), 4),
                graph_risk=round(float(graph_risk[i]), 4), rule_risk=round(r_rule, 4),
                flagged=bool(proba[i] >= self.pipe.threshold or s >= 0.55),
                reasons=reasons[:6], top_features=expl[i]))
        return results
