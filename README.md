# 🛡️ FraudNet — Graph-Augmented Credit Card Fraud Detection

An end-to-end fraud detection system on the **IEEE-CIS Fraud Detection** dataset that combines a
transaction-level **XGBoost** classifier with a **heterogeneous entity graph** (cards, devices,
e-mail domains, addresses, IP proxies) to detect organised **fraud rings**, served through a
**FastAPI** backend and a **Streamlit** investigation dashboard with explainable risk scores.

[![CI](https://github.com/arihxnt25/Fraud_detection/actions/workflows/ci.yml/badge.svg)](../../actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Why a graph?

A tabular model scores each transaction in isolation. A fraud ring defeats that by making every
individual transaction look ordinary — normal product, normal amount, mainstream e-mail. What gives
the ring away is **structure**: dozens of "different" cards sharing one device fingerprint and one
billing address. FraudNet turns that structure into features, ranked clusters and evidence an
analyst can act on.

```
            ┌────────────┐   uses    ┌──────────────┐
  card A ───┤            ├──────────►│ device: X    │◄──── card B, card C, card D …
            │ connected_ │           └──────────────┘
            │  through   │ occurs_at ┌──────────────┐
  card B ───┤ (≥2 shared ├──────────►│ address: 204 │◄──── card C, card D …
            │  entities) │           └──────────────┘
            └────────────┘
```

## Architecture

```
IEEE-CIS CSVs ─► loader (memory-reduced, stratified sampling, out-of-time split)
                    │
        ┌───────────┴──────────────┐
        ▼                          ▼
 FeatureBuilder              FraudGraph (NetworkX)
 • card_uid reconstruction   • card ↔ entity bipartite graph (typed edges)
 • time / amount / velocity  • card–card projection: linked only if ≥2 shared entities
 • frequency & uid aggregates• connected components → fraud-ring clusters
                             • personalised PageRank = "fraud proximity"
                             • Bayesian-smoothed, leave-one-out entity fraud rates
        └───────────┬──────────────┘
                    ▼
        XGBoost (out-of-fold graph features, PR-AUC early stopping)
                    ▼
        RiskScorer: 0.7·model + 0.2·graph + 0.1·rules → tier + reasons + TreeSHAP
                    ▼
     FastAPI (/score, /clusters, /graph, /cases)  ·  Streamlit dashboard  ·  MongoDB
```

See [`docs/architecture.md`](docs/architecture.md) for design decisions.

## Key engineering decisions

| Problem | Decision |
|---|---|
| No card-holder ID in IEEE-CIS | Reconstruct `card_uid = card1 + addr1 + (day − D1)` (approach from the competition's top solutions) |
| Random splits leak a card's future into training | **Out-of-time** split: train on earliest 80%, validate on latest 20% |
| Graph features from training labels leak into training rows | **Out-of-fold** graph features (GroupKFold by card) + **leave-one-out** at serving |
| One fraud card on a rare device ⇒ "100% fraud device" | **Bayesian smoothing** toward the global prior |
| Hubs (`gmail.com`, popular phone models) merge everything into one component | Cards link only when they share **≥ 2 distinct** non-hub entities; hub cutoff adapts to population size |
| Analysts must justify every decline | Per-transaction **TreeSHAP** contributions + plain-English reasons |
| 590K rows × 434 cols on a laptop | dtype downcasting, curated V-column subset, 10% stratified sampling for dev, full run on Colab |

## Quick start

```bash
git clone https://github.com/arihxnt25/Fraud_detection.git
cd Fraud_detection
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1) Demo — synthetic IEEE-CIS-shaped data with planted fraud rings (~1 min, no download)
python scripts/run_pipeline.py --synthetic

# 2) Real data — put train_transaction.csv + train_identity.csv in data/raw/
python scripts/run_pipeline.py --sample 0.1     # laptop: 10% stratified sample
python scripts/run_pipeline.py --sample 1.0     # full data (Colab: notebooks/colab_full_training.ipynb)

# 3) Serve
uvicorn api.main:app --reload --port 8000       # docs at http://localhost:8000/docs
streamlit run dashboard/app.py                  # dashboard at http://localhost:8501

# Or everything (API + dashboard + MongoDB) with Docker
docker compose up --build
```

Dataset: <https://www.kaggle.com/c/ieee-fraud-detection/data> (accept the competition rules, then
`kaggle competitions download -c ieee-fraud-detection`).

## Results

### Synthetic demo data (30K transactions, 12 planted rings, out-of-time validation)

| Metric | Tabular only | Graph-augmented |
|---|---|---|
| ROC-AUC | 0.964 | 0.964 |
| PR-AUC | 0.847 | 0.847 |
| Precision @ best-F1 threshold | 0.924 | **0.941** |
| Recall @ best-F1 threshold | 0.730 | 0.710 |
| Fraud rings recovered | — | **12 / 12 (100% fraud each)** |

Honest reading: on this synthetic data, frequency encodings such as `DeviceInfo_freq` already
capture most of the shared-device signal, so overall AUC is unchanged. The graph's distinct value is
**ring discovery and investigation** (whole clusters surfaced, with the shared entities as evidence)
and higher precision. Graph features are nonetheless the top two by gain
(`g_fraud_proximity`, `g_component_size`).

### IEEE-CIS (real data)

Run `python scripts/run_pipeline.py --sample 1.0` (or the Colab notebook) and paste
`artifacts/report.json` here. The script prints the same tabular-vs-graph ablation table.

| Metric | Tabular only | Graph-augmented |
|---|---|---|
| ROC-AUC | _run to fill_ | _run to fill_ |
| PR-AUC | _run to fill_ | _run to fill_ |

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | model + graph summary, storage backend |
| `POST` | `/score` | score one transaction → score, tier, reasons, top SHAP features |
| `POST` | `/score/batch` | up to 5,000 transactions |
| `GET` | `/alerts?min_tier=HIGH` | recently scored high-risk transactions |
| `GET` | `/clusters` | ranked fraud-ring clusters |
| `GET` | `/graph/{node}` | ego network, e.g. `/graph/device:SM-G960U` |
| `GET` | `/model/importance`, `/model/metrics` | model introspection |
| `POST/PATCH/GET` | `/cases` | analyst case management |

A **brand-new card** (never seen in training) making an unremarkable $52.90 purchase — but on a
device fingerprint and billing address shared by a known ring (real output on the demo model):

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "TransactionDT": 13220889, "TransactionAmt": 52.9, "ProductCD": "H", "card1": 99001,
  "addr1": 161.0, "D1": 65.0, "P_emaildomain": "aol.com",
  "DeviceInfo": "Moto G (5) Plus Build/R01978"}'
```

```json
{
  "risk_score": 0.9487, "risk_tier": "CRITICAL", "flagged": true,
  "model_probability": 0.9986, "graph_risk": 0.9271, "rule_risk": 0.0,
  "reasons": ["member of fraud cluster (size 55, ~93% fraud)",
              "device previously used by fraudulent cards"],
  "top_features": [
    {"feature": "DeviceInfo_freq",  "value": 0.0023, "contribution": 3.08},
    {"feature": "g_component_size", "value": 55.0,   "contribution": 1.80},
    {"feature": "C1",               "value": null,   "contribution": 1.47}]
}
```

## Dashboard

Five tabs: **Overview** (metrics, ablation, feature importance) · **Alert queue** (tiered, with
reasons and ground truth) · **Fraud rings** (cluster table + interactive network) · **Graph
explorer** (ego network of any card/device/e-mail/address) · **Score a transaction** (what-if
editing with TreeSHAP bars).

## Project structure

```
├── api/main.py                  FastAPI service
├── dashboard/app.py             Streamlit investigation console
├── scripts/run_pipeline.py      train + evaluate + ablation + artifacts
├── src/fraudnet/
│   ├── config.py                paths, entity schema, hyper-parameters
│   ├── data/loader.py           IEEE-CIS loading, memory reduction, sampling, time split
│   ├── data/synthetic.py        IEEE-CIS-shaped generator with planted rings
│   ├── features/engineering.py  card_uid, temporal, amount, velocity, encodings
│   ├── graph/network.py         heterogeneous graph, rings, proximity, graph features
│   ├── models/pipeline.py       FraudPipeline (features + graph + XGBoost), metrics
│   ├── risk/scoring.py          blended explainable risk score
│   └── db/store.py              MongoDB store with in-memory fallback
├── notebooks/                   Colab full-data training
├── tests/                       16 pytest tests (core + API)
├── docs/architecture.md
├── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Testing

```bash
pytest -q          # 16 tests: features, leakage guards, ring recovery, persistence, API
ruff check .
```

## Limitations & next steps

- The graph is rebuilt in batch; a streaming version would update entity counts incrementally
  (e.g. Neo4j or a feature store).
- Ring detection uses connected components; Louvain/Leiden community detection would split large
  mixed components more finely.
- A GNN (GraphSAGE on the card–entity graph) is the natural next model to compare against.
- Synthetic results are a pipeline demo, not a performance claim — report IEEE-CIS numbers.

## License

MIT
