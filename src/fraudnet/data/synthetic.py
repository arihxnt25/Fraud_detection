"""Synthetic IEEE-CIS-shaped data with planted fraud rings.

Lets the whole pipeline (training, graph analysis, API, dashboard, CI tests)
run without the 1.3 GB Kaggle download. Fraud rings share devices, e-mail
domains and billing addresses across many cards -- exactly the pattern the
graph module is designed to surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["W", "C", "R", "H", "S"]
CARD4 = ["visa", "mastercard", "american express", "discover"]
CARD6 = ["debit", "credit"]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
          "outlook.com", "icloud.com", "comcast.net", "protonmail.com", "mail.com"]
DEVICES = ["Windows", "iOS Device", "MacOS", "SM-G960U", "Trident/7.0", "SM-J700M",
           "Moto G (5) Plus", "rv:11.0", "Linux", "Pixel 3"]
PROXY = ["IP_PROXY:TRANSPARENT", "IP_PROXY:ANONYMOUS", "IP_PROXY:HIDDEN"]


def _mask(cond, values):
    """np.where(cond, values, NaN) that works for string arrays on NumPy 2.x."""
    out = np.asarray(values, dtype=object).copy()
    out[~np.asarray(cond)] = np.nan
    return out


def make_synthetic(n: int = 20_000, fraud_rate: float = 0.035, n_rings: int | None = None,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rings = n_rings or max(12, n // 2_500)  # rings stay ~30-80 transactions each
    n_cards = max(n // 8, 50)
    card_pool = pd.DataFrame({
        "card1": rng.integers(1000, 18000, n_cards),
        "card2": rng.choice(np.arange(100, 600), n_cards).astype(float),
        "card4": rng.choice(CARD4, n_cards, p=[.62, .32, .03, .03]),
        "card6": rng.choice(CARD6, n_cards, p=[.75, .25]),
        "addr1": rng.integers(100, 540, n_cards).astype(float),
        "home_email": rng.choice(EMAILS, n_cards, p=[.4, .17, .08, .07, .05, .08, .06, .04, .03, .02]),
        "home_device": [f"{d}" for d in rng.choice(DEVICES, n_cards)],
        "first_day": rng.integers(-400, 120, n_cards),
    })

    idx = rng.integers(0, n_cards, n)
    df = card_pool.iloc[idx].reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(2_987_000, 2_987_000 + n))
    df.insert(1, "TransactionDT", np.sort(rng.integers(86_400, 86_400 * 182, n)))
    df["TransactionAmt"] = np.round(rng.lognormal(3.9, 1.0, n), 2)
    df["ProductCD"] = rng.choice(PRODUCTS, n, p=[.74, .12, .06, .06, .02])
    df["P_emaildomain"] = df.pop("home_email")
    df["R_emaildomain"] = _mask(rng.random(n) < .25, rng.choice(EMAILS, n))
    has_id = rng.random(n) < .3
    df["DeviceInfo"] = _mask(has_id, df.pop("home_device"))
    df["DeviceType"] = _mask(has_id, rng.choice(["desktop", "mobile"], n))
    df["id_23"] = _mask(has_id & (rng.random(n) < .05), rng.choice(PROXY, n))
    df["id_30"] = _mask(has_id, rng.choice(["Windows 10", "iOS 12", "Android 8", "Mac OS X"], n))
    df["id_31"] = _mask(has_id, rng.choice(["chrome 70.0", "mobile safari 12.0", "edge 17.0"], n))
    df["dist1"] = np.where(rng.random(n) < .4, rng.exponential(30, n), np.nan)
    # D1 = days since the card was first seen -> card1+addr1+(day-D1) is stable per card
    df["D1"] = np.clip(df["TransactionDT"] // 86_400 - df.pop("first_day"), 0, None).astype(float)
    df["D15"] = df["D1"] + rng.integers(0, 50, n)
    for i in range(1, 15):
        df[f"C{i}"] = rng.poisson(1.5, n).astype(float)
    df["M4"] = rng.choice(["M0", "M1", "M2", None], n)
    df["M6"] = rng.choice(["T", "F", None], n)
    for v in (1, 3, 13, 20, 45, 87, 258, 294, 307, 317):
        df[f"V{v}"] = rng.gamma(1.0, 1.0, n)
    df["isFraud"] = 0

    # --- background (non-ring) fraud: noisy, weakly separable ---------------
    n_bg = int(n * fraud_rate * 0.45)
    bg = rng.choice(n, n_bg, replace=False)
    df.loc[bg, "isFraud"] = 1
    df.loc[bg, "TransactionAmt"] *= rng.uniform(1.0, 3.5, n_bg)
    df.loc[bg, "ProductCD"] = rng.choice(["C", "W"], n_bg)
    df.loc[bg, "C1"] += rng.poisson(3, n_bg)

    # --- fraud rings: many cards sharing device / email / address ------------
    n_ring_tx = int(n * fraud_rate) - n_bg
    ring_rows = rng.choice(np.setdiff1d(np.arange(n), bg), n_ring_tx, replace=False)
    ring_ids = rng.integers(0, n_rings, n_ring_tx)
    for r in range(n_rings):
        rows = ring_rows[ring_ids == r]
        if len(rows) == 0:
            continue
        df.loc[rows, "isFraud"] = 1
        # rings look ordinary transaction-by-transaction (normal products,
        # mainstream e-mail, typical amounts); what gives them away is that
        # many "different" cards share one device fingerprint + billing address
        df.loc[rows, "DeviceInfo"] = f"{rng.choice(DEVICES)} Build/R{r:02d}{rng.integers(100, 999)}"
        df.loc[rows, "DeviceType"] = rng.choice(["desktop", "mobile"])
        df.loc[rows, "P_emaildomain"] = rng.choice(EMAILS[:6])
        df.loc[rows, "addr1"] = float(rng.integers(100, 540))
        if rng.random() < 0.5:
            df.loc[rows, "id_23"] = rng.choice(PROXY)
        df.loc[rows, "id_30"] = rng.choice(["Windows 10", "iOS 12", "Android 8"])
        # rings stay active for weeks, so many straddle the train/valid split
        span = 86_400 * int(rng.integers(20, 70))
        start = rng.integers(86_400, 86_400 * 182 - span)
        df.loc[rows, "TransactionDT"] = start + rng.integers(0, span, len(rows))

        # ring cards are fresh cards opened just before the ring became active
        df.loc[rows, "D1"] = ((df.loc[rows, "TransactionDT"] - start) // 86_400).astype(float)

    return df.sort_values("TransactionDT").reset_index(drop=True)
