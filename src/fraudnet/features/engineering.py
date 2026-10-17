"""Feature engineering: temporal, behavioural and card-identity features.

The key idea (from the IEEE-CIS winning solutions) is that the dataset has no
explicit card-holder ID, but ``card1 + addr1 + (day - D1)`` reconstructs one
quite reliably. Fraud labels are sticky per card holder, so aggregations over
this ``card_uid`` are the strongest behavioural signal.

``FeatureBuilder`` follows the fit/transform pattern so the exact statistics
learned on training data are reused at inference time (API / dashboard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraudnet.config import CATEGORICAL_COLS, ID_COL, TARGET

SECONDS_PER_DAY = 86_400
UID_AGG_COLS = ["TransactionAmt", "dist1", "C1", "C13", "D15"]


def add_card_uid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    day = df["TransactionDT"] // SECONDS_PER_DAY
    d1 = df["D1"] if "D1" in df else 0
    df["card_uid"] = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
                      + (day - d1).astype(str))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["TransactionDT"]
    df["tx_hour"] = (t // 3600) % 24
    df["tx_dow"] = (t // SECONDS_PER_DAY) % 7
    df["tx_is_night"] = df["tx_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amt = df["TransactionAmt"]
    df["amt_log"] = np.log1p(amt)
    df["amt_cents"] = np.round(amt - np.floor(amt), 3)
    df["amt_is_round"] = (df["amt_cents"] == 0).astype(int)
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time since the same card / device was last seen (within the frame)."""
    df = df.sort_values("TransactionDT").copy()
    for key, name in (("card_uid", "uid"), ("DeviceInfo", "device")):
        if key in df:
            prev = df.groupby(key, dropna=True)["TransactionDT"].shift(1)
            df[f"secs_since_prev_{name}"] = (df["TransactionDT"] - prev).astype(float)
    df["uid_tx_rank"] = df.groupby("card_uid").cumcount()
    return df.sort_index()


class FeatureBuilder:
    """Stateful transformer: learns encodings/aggregates on fit, applies on transform."""

    def __init__(self, categorical_cols: list[str] | None = None):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.freq_maps: dict[str, dict] = {}
        self.cat_maps: dict[str, dict] = {}
        self.uid_stats: pd.DataFrame | None = None
        self.feature_names: list[str] = []

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        df = self._base(df)
        for col in self.categorical_cols + ["card1", "addr1", "card_uid"]:
            if col in df:
                self.freq_maps[col] = df[col].value_counts(dropna=False, normalize=True).to_dict()
        for col in self.categorical_cols:
            if col in df:
                cats = df[col].astype("string").fillna("__nan__").unique()
                self.cat_maps[col] = {c: i for i, c in enumerate(sorted(cats))}
        aggs = {c: ["mean", "std"] for c in UID_AGG_COLS if c in df}
        stats = df.groupby("card_uid").agg(aggs)
        stats.columns = [f"uid_{c}_{s}" for c, s in stats.columns]
        stats["uid_tx_count"] = df.groupby("card_uid").size()
        self.uid_stats = stats
        self.feature_names = list(self._finish(df).columns)
        return self

    # ---------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.uid_stats is None:
            raise RuntimeError("FeatureBuilder must be fit before transform")
        X = self._finish(self._base(df))
        return X.reindex(columns=self.feature_names)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        df = add_card_uid(df)
        df = add_time_features(df)
        df = add_amount_features(df)
        return add_velocity_features(df)

    def _finish(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, fmap in self.freq_maps.items():
            if col in df:
                df[f"{col}_freq"] = df[col].map(fmap).astype(float).fillna(0.0)
        for col, cmap in self.cat_maps.items():
            if col in df:
                df[col] = df[col].astype("string").fillna("__nan__").map(cmap).fillna(-1).astype(int)
        if self.uid_stats is not None:
            df = df.join(self.uid_stats, on="card_uid")
            df["uid_tx_count"] = df["uid_tx_count"].fillna(0)
            if "uid_TransactionAmt_mean" in df:
                df["amt_vs_uid_mean"] = df["TransactionAmt"] / (df["uid_TransactionAmt_mean"] + 1e-3)
        drop = {TARGET, ID_COL, "TransactionDT", "card_uid"}
        X = df.drop(columns=[c for c in drop if c in df])
        X = X.select_dtypes(include=[np.number, "bool"])
        return X.astype(np.float32)
