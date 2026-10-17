"""Central configuration: paths, column groups and model hyper-parameters."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FRAUDNET_DATA_DIR", ROOT / "data" / "raw"))
ARTIFACT_DIR = Path(os.getenv("FRAUDNET_ARTIFACT_DIR", ROOT / "artifacts"))

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"

# Entity columns that become graph nodes. Each maps a node type to the raw
# column(s) that identify it in the IEEE-CIS schema.
ENTITY_COLUMNS: dict[str, list[str]] = {
    "card": ["card_uid"],          # engineered: card1 + addr1 + D1-normalised
    "device": ["DeviceInfo"],
    "email_payer": ["P_emaildomain"],
    "email_recv": ["R_emaildomain"],
    "address": ["addr1"],
    "ip_proxy": ["id_23"],          # IP proxy type (anonymised in IEEE-CIS)
}

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M4", "M6", "DeviceType", "DeviceInfo", "id_23", "id_30", "id_31",
]


@dataclass
class ModelConfig:
    n_estimators: int = 600
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    min_child_weight: int = 3
    early_stopping_rounds: int = 50
    eval_metric: str = "aucpr"
    tree_method: str = "hist"
    random_state: int = 42
    extra: dict = field(default_factory=dict)

    def as_xgb_params(self) -> dict:
        p = {k: v for k, v in self.__dict__.items() if k != "extra"}
        p.update(self.extra)
        return p


@dataclass
class RiskConfig:
    """Weights used to blend model probability and graph risk into one score."""
    w_model: float = 0.70
    w_graph: float = 0.20
    w_rules: float = 0.10
    tiers: tuple = ((0.80, "CRITICAL"), (0.55, "HIGH"), (0.30, "MEDIUM"), (0.0, "LOW"))
