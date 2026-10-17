"""End-to-end model: tabular features + graph features -> XGBoost.

``FraudPipeline`` is the single object that is trained, saved, and loaded by
the API and dashboard, so inference uses exactly the training-time encodings.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from fraudnet.config import ARTIFACT_DIR, TARGET, ModelConfig
from fraudnet.features.engineering import FeatureBuilder, add_card_uid
from fraudnet.graph.network import FraudGraph

log = logging.getLogger(__name__)


class FraudPipeline:
    def __init__(self, model_cfg: ModelConfig | None = None, use_graph: bool = True,
                 max_entity_degree: int | None = None, oof_folds: int = 5):
        self.cfg = model_cfg or ModelConfig()
        self.use_graph = use_graph
        self.max_entity_degree = max_entity_degree
        self.oof_folds = oof_folds
        self.features = FeatureBuilder()
        self.graph = FraudGraph(max_entity_degree=max_entity_degree) if use_graph else None
        self.model: xgb.XGBClassifier | None = None
        self.threshold: float = 0.5
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    # ---------------------------------------------------------------- matrix
    def _matrix(self, df: pd.DataFrame, graph_feats: pd.DataFrame | None = None) -> pd.DataFrame:
        X = self.features.transform(df)
        if self.graph is not None:
            g = graph_feats if graph_feats is not None else self.graph.transform(df)
            X = pd.concat([X, g.set_index(X.index)], axis=1)
        return X

    def _oof_graph_features(self, train: pd.DataFrame) -> pd.DataFrame:
        """Out-of-fold graph features for the training rows.

        If training rows were scored by a graph built from *all* training
        labels, ring members would see their ring-mates' labels and the model
        would learn "graph rate high => fraud" far too strongly -- features
        that look near-perfect in training but are much weaker on new data.
        Folds are grouped by card so a card never informs its own features.
        """
        parts = []
        gkf = GroupKFold(n_splits=self.oof_folds)
        for fit_idx, oof_idx in gkf.split(train, groups=train["card_uid"]):
            g = FraudGraph(max_entity_degree=self.max_entity_degree).fit(train.iloc[fit_idx])
            parts.append(g.transform(train.iloc[oof_idx]))
        return pd.concat(parts).loc[train.index]

    # ------------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
        t0 = time.time()
        train, valid = add_card_uid(train), add_card_uid(valid)
        self.features.fit(train)
        oof = None
        if self.graph is not None:
            oof = self._oof_graph_features(train) if self.oof_folds > 1 else None
            self.graph.fit(train)  # serving graph: labels from the training window only
        Xtr, Xva = self._matrix(train, oof), self._matrix(valid)
        ytr, yva = train[TARGET].values, valid[TARGET].values
        self.feature_names = list(Xtr.columns)

        pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        params = self.cfg.as_xgb_params()
        params["scale_pos_weight"] = min(pos_weight, 25.0)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        p = self.model.predict_proba(Xva)[:, 1]
        self.threshold = best_f1_threshold(yva, p)
        self.metrics = evaluate(yva, p, self.threshold)
        self.metrics.update(n_train=len(train), n_valid=len(valid), use_graph=self.use_graph,
                            n_features=len(self.feature_names),
                            best_iteration=int(getattr(self.model, "best_iteration", 0) or 0),
                            train_seconds=round(time.time() - t0, 1))
        log.info("Validation: %s", {k: self.metrics[k] for k in ("roc_auc", "pr_auc", "f1")})
        return self.metrics

    # ------------------------------------------------------------- inference
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._matrix(add_card_uid(df)))[:, 1]

    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Per-row top feature contributions (TreeSHAP via XGBoost pred_contribs)."""
        X = self._matrix(add_card_uid(df))
        contribs = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(contribs[i]))[:top_k]
            out.append([{"feature": self.feature_names[j], "value": _num(X.iat[i, j]),
                         "contribution": round(float(contribs[i, j]), 4)} for j in order])
        return out

    def feature_importance(self, top_k: int = 25) -> pd.DataFrame:
        gain = self.model.get_booster().get_score(importance_type="gain")
        imp = pd.DataFrame({"feature": list(gain), "gain": list(gain.values())})
        imp["group"] = np.where(imp.feature.str.startswith("g_"), "graph", "tabular")
        return imp.sort_values("gain", ascending=False).head(top_k).reset_index(drop=True)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path | str = ARTIFACT_DIR / "fraud_pipeline.joblib") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        (path.parent / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        return path

    @staticmethod
    def load(path: Path | str = ARTIFACT_DIR / "fraud_pipeline.joblib") -> FraudPipeline:
        return joblib.load(path)


# ------------------------------------------------------------------ metrics
def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def recall_at_precision(y, p, target_precision: float = 0.9) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec[prec >= target_precision]
    return float(ok.max()) if len(ok) else 0.0


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "recall_at_90_precision": round(recall_at_precision(y, p, 0.9), 4),
        "fraud_rate_valid": round(float(np.mean(y)), 4),
    }


def _num(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 4)
