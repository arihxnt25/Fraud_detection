# Architecture & design notes

## 1. Data layer (`fraudnet.data`)

- **Loading.** `train_transaction.csv` (590,540 × 394) is merged with `train_identity.csv`
  (144,233 × 41) on `TransactionID`. Numeric columns are downcast (float64 → float32, ints to the
  smallest type) and a curated subset of ~125 of the 339 `V*` columns is kept, cutting memory by
  roughly 60%.
- **Sampling.** `--sample 0.1` draws a per-class stratified sample (fraud rate preserved exactly)
  for development on machines with limited RAM; the full run is intended for Colab.
- **Split.** Out-of-time: data is sorted by `TransactionDT`, the first 80% trains, the last 20%
  validates. This mirrors production (train on the past, score the future) and is the setting in
  which IEEE-CIS public/private leaderboards were computed.

## 2. Features (`fraudnet.features`)

| Group | Examples |
|---|---|
| Identity | `card_uid = card1_addr1_(day − D1)` — `D1` is "days since card first used", so `day − D1` is the card's start date |
| Temporal | hour of day, day of week, night flag |
| Amount | log amount, cents part, round-amount flag, amount ÷ card's mean |
| Velocity | seconds since the card's / device's previous transaction, card transaction rank |
| Encodings | frequency encodings, label encodings learned on training data only |
| Card aggregates | mean/std of amount, `dist1`, `C1`, `C13`, `D15` per `card_uid`; transaction count |

`FeatureBuilder` is fit on training data and reused verbatim at inference (unseen categories map
to `-1` / frequency `0`).

## 3. Graph (`fraudnet.graph.network.FraudGraph`)

**Schema.** Nodes: `card`, `device` (DeviceInfo), `email_payer`, `email_recv`, `address` (addr1),
`ip_proxy` (id_23). Edges: `card —uses→ device/email/ip_proxy`, `card —occurs_at→ address`
(weighted by transaction count), and `card —connected_through→ card`.

**Ring detection.** Two cards are joined by `connected_through` only when they share **at least
two distinct non-hub entities**. A single shared attribute is common coincidence (millions use
Gmail; thousands own the same phone model). Hubs are entities above an adaptive degree cutoff
(1% of cards, clipped to [100, 1000]). Connected components of the card–card graph are candidate
rings, ranked by number of fraud cards and fraud rate.

**Features per transaction**

| Feature | Meaning |
|---|---|
| `g_<entity>_n_cards` | how many cards use this device / e-mail / address / proxy |
| `g_<entity>_fraud_rate` | smoothed fraud rate of those cards (own card excluded) |
| `g_max_entity_fraud_rate` | worst of the above |
| `g_component_size`, `g_component_fraud_rate` | the card's ring cluster |
| `g_fraud_proximity` | personalised PageRank seeded on known fraud cards |

**Unseen cards** (first appearance at scoring time) join a ring if ≥2 of their entities belong to
that ring's component, and take proximity from their entities.

## 4. Leakage controls

1. **Out-of-time validation** — only past labels inform graph features for the validation window.
2. **Out-of-fold training features** — training rows get graph features from a graph built on the
   *other* folds (GroupKFold by `card_uid`). Without this, ring members see their ring-mates'
   labels in training, the model over-trusts graph features, and validation performance drops
   (observed during development: ROC-AUC fell from 0.96 to 0.77 before this fix).
3. **Leave-one-out** at serving — a training card never sees its own label via the graph.
4. **Bayesian smoothing** — `(fraud + prior·m) / (cards + m)` with `m = 5` prevents tiny entities
   from reporting 0% or 100%.

## 5. Model

XGBoost (`hist`, depth 8, lr 0.05, up to 600 trees, early stopping on validation PR-AUC,
`scale_pos_weight` capped at 25). Decision threshold = best-F1 point on validation. Metrics: ROC-AUC,
PR-AUC (the honest metric at ~3.5% prevalence), precision/recall/F1, recall at 90% precision.

## 6. Risk scoring

```
score = 0.7 · p_model + 0.2 · graph_risk + 0.1 · rule_risk
score = max(score, 0.95 · p_model, 0.85 · graph_risk if graph_risk > 0.6)
tiers: CRITICAL ≥ 0.80 > HIGH ≥ 0.55 > MEDIUM ≥ 0.30 > LOW
```

Rules are deliberately simple and auditable (high amount, large night purchase, anonymous proxy,
high-risk e-mail domain). Each result carries plain-English reasons and the top-5 TreeSHAP
contributions (`pred_contribs=True`, no extra SHAP dependency).

## 7. Serving

- **FastAPI** loads one serialized `FraudPipeline` (features + graph + model), so training and
  inference are identical by construction. Pydantic validates inputs; extra IEEE-CIS fields are
  accepted.
- **Storage** — MongoDB (`MONGO_URI`) for scored transactions and analyst cases, with an in-memory
  fallback so the API runs with zero setup.
- **Dashboard** — Streamlit + Plotly; network layouts via NetworkX spring layout.
