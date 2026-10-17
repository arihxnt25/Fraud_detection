"""Train and evaluate the fraud pipeline.

Examples:
    # demo on synthetic data (no download needed, ~1 min)
    python scripts/run_pipeline.py --synthetic

    # laptop: 10% stratified sample of IEEE-CIS
    python scripts/run_pipeline.py --sample 0.1

    # Colab / big machine: full dataset
    python scripts/run_pipeline.py --sample 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudnet.config import ARTIFACT_DIR, DATA_DIR  # noqa: E402
from fraudnet.data.loader import load_ieee_cis, time_split  # noqa: E402
from fraudnet.data.synthetic import make_synthetic  # noqa: E402
from fraudnet.models.pipeline import FraudPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated demo data")
    ap.add_argument("--n", type=int, default=30_000, help="rows for synthetic data")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--sample", type=float, default=None, help="stratified sample fraction")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--no-ablation", action="store_true", help="skip the tabular-only baseline")
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = make_synthetic(args.n) if args.synthetic else load_ieee_cis(args.data_dir, args.sample)
    train, valid = time_split(df, args.valid_frac)
    logging.info("train=%s valid=%s fraud%%=%.2f", train.shape, valid.shape, 100 * df.isFraud.mean())

    report = {}
    if not args.no_ablation:
        base = FraudPipeline(use_graph=False)
        report["tabular_only"] = base.fit(train, valid)

    pipe = FraudPipeline(use_graph=True)
    report["graph_augmented"] = pipe.fit(train, valid)
    pipe.save(out / "fraud_pipeline.joblib")

    # artefacts consumed by the dashboard
    valid.to_parquet(out / "valid_sample.parquet") if len(valid) <= 200_000 else \
        valid.sample(200_000, random_state=0).to_parquet(out / "valid_sample.parquet")
    pipe.feature_importance(30).to_csv(out / "feature_importance.csv", index=False)
    clusters = [c.to_dict() for c in pipe.graph.fraud_clusters(top_k=100)]
    (out / "fraud_clusters.json").write_text(json.dumps(clusters, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Validation (out-of-time) ===")
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1", "recall_at_90_precision"]
    print(f"{'metric':<24}" + "".join(f"{k:>18}" for k in report))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{report[m][k]:>18}" for m in report))
    print(f"\nSuspicious clusters found: {len(clusters)}")
    print(f"Artifacts written to {out}")


if __name__ == "__main__":
    main()
