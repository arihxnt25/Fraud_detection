"""Load the IEEE-CIS Fraud Detection dataset (Kaggle) with memory-aware options.

Expected files in ``DATA_DIR``:
    train_transaction.csv, train_identity.csv
(download: https://www.kaggle.com/c/ieee-fraud-detection/data)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraudnet.config import DATA_DIR, TARGET

log = logging.getLogger(__name__)

# The V-columns are 339 heavily correlated features; keeping a representative
# subset cuts RAM by ~60% with negligible AUC loss on this dataset.
V_KEEP = [f"V{i}" for i in (1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40,
                            41, 44, 47, 48, 54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80,
                            82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123,
                            124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 162, 165,
                            160, 166, 178, 176, 173, 182, 187, 203, 205, 207, 215, 169,
                            171, 175, 180, 185, 188, 198, 210, 209, 218, 223, 224, 226,
                            228, 229, 235, 240, 258, 257, 253, 252, 260, 261, 264, 266,
                            267, 274, 277, 220, 221, 234, 238, 250, 271, 294, 284, 285,
                            286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289,
                            296, 301, 314, 332, 325, 335, 338)]


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe dtype."""
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def load_ieee_cis(data_dir: Path | str = DATA_DIR, sample_frac: float | None = None,
                  random_state: int = 42, keep_all_v: bool = False) -> pd.DataFrame:
    """Load and merge transaction + identity tables.

    Args:
        sample_frac: optional stratified sample fraction (e.g. 0.1 for laptops
            with limited RAM; use 1.0 / None on Colab for the full data).
    """
    data_dir = Path(data_dir)
    tx_path, id_path = data_dir / "train_transaction.csv", data_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found. Download IEEE-CIS from Kaggle into {data_dir} "
            "or run `python scripts/run_pipeline.py --synthetic` for a demo.")

    header = pd.read_csv(tx_path, nrows=0).columns
    usecols = [c for c in header if not c.startswith("V") or keep_all_v or c in V_KEEP]
    tx = reduce_memory(pd.read_csv(tx_path, usecols=usecols))
    log.info("Loaded transactions: %s", tx.shape)

    if sample_frac and sample_frac < 1.0:
        tx = stratified_sample(tx, sample_frac, random_state)
        log.info("Stratified sample (%.0f%%): %s", sample_frac * 100, tx.shape)

    if id_path.exists():
        ident = pd.read_csv(id_path)
        ident.columns = [c.replace("-", "_") for c in ident.columns]  # test files use id-01
        tx = tx.merge(reduce_memory(ident), on="TransactionID", how="left")
    return tx.sort_values("TransactionDT").reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, frac: float, random_state: int = 42) -> pd.DataFrame:
    # per-class sampling keeps the fraud rate exact (and avoids groupby.apply,
    # which drops the grouping column in pandas >= 3)
    parts = [df[df[TARGET] == k].sample(frac=frac, random_state=random_state)
             for k in df[TARGET].unique()]
    return pd.concat(parts).sort_values("TransactionDT").reset_index(drop=True)


def time_split(df: pd.DataFrame, valid_frac: float = 0.2, time_col: str = "TransactionDT"):
    """Out-of-time split: train on the earliest data, validate on the latest.

    Random splits leak future behaviour of the same card into training and
    inflate AUC on IEEE-CIS by several points.
    """
    df = df.sort_values(time_col)
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def fraud_rate(df: pd.DataFrame) -> float:
    return float(np.mean(df[TARGET])) if TARGET in df else float("nan")
