import numpy as np
import pandas as pd
import pytest

from fraudnet.features.engineering import FeatureBuilder, add_card_uid
from fraudnet.graph.network import FraudGraph
from fraudnet.models.pipeline import FraudPipeline
from fraudnet.risk.scoring import RiskScorer


def test_synthetic_shape(data):
    assert {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1"} <= set(data.columns)
    assert 0.02 < data.isFraud.mean() < 0.06


def test_time_split_is_out_of_time(split):
    train, valid = split
    assert train.TransactionDT.max() <= valid.TransactionDT.min()


def test_feature_builder_consistent_columns(split):
    train, valid = split
    fb = FeatureBuilder().fit(train)
    Xtr, Xva = fb.transform(train), fb.transform(valid)
    assert list(Xtr.columns) == list(Xva.columns)
    assert "isFraud" not in Xtr.columns
    assert Xtr.dtypes.eq(np.float32).all()


def test_unseen_categories_do_not_crash(split):
    train, valid = split
    fb = FeatureBuilder().fit(train)
    odd = valid.head(3).copy()
    odd["ProductCD"] = "ZZZ"
    odd["P_emaildomain"] = "never-seen.org"
    assert fb.transform(odd).shape[0] == 3


def test_graph_finds_planted_rings(split):
    train, _ = split
    g = FraudGraph().fit(add_card_uid(train))
    clusters = g.fraud_clusters(min_cards=3, min_fraud_rate=0.5)
    assert len(clusters) >= 3
    assert all(c.fraud_rate >= 0.5 for c in clusters)


def test_graph_features_leave_one_out(split):
    """A lone fraud card must not see its own label through the graph."""
    train, _ = split
    one = train.head(1).copy()
    one["isFraud"] = 1
    one["DeviceInfo"] = "unique-device-xyz"
    one["card1"] = 99999
    df = add_card_uid(pd.concat([train, one], ignore_index=True))
    g = FraudGraph().fit(df)
    f = g.transform(df.tail(1))
    # with its own label removed the card only sees the (smoothed) prior
    assert f["g_device_fraud_rate"].iloc[0] == pytest.approx(g.prior)
    assert f["g_component_fraud_rate"].iloc[0] == pytest.approx(g.prior)


def test_pipeline_quality(trained):
    pipe, _ = trained
    assert pipe.metrics["roc_auc"] > 0.85
    assert any(f.startswith("g_") for f in pipe.feature_names)


def test_save_load_roundtrip(trained, split):
    pipe, path = trained
    _, valid = split
    loaded = FraudPipeline.load(path)
    np.testing.assert_allclose(pipe.predict_proba(valid.head(50)), loaded.predict_proba(valid.head(50)))


def test_risk_scores_are_explained(trained, split):
    pipe, _ = trained
    _, valid = split
    res = RiskScorer(pipe).score(valid.head(20))
    assert len(res) == 20
    for r in res:
        assert 0 <= r.risk_score <= 1
        assert r.risk_tier in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert len(r.top_features) == 5
    worst = max(res, key=lambda r: r.risk_score)
    if worst.risk_tier in {"HIGH", "CRITICAL"}:
        assert worst.reasons


def test_stratified_sample_keeps_rate(data):
    from fraudnet.data.loader import stratified_sample
    s = stratified_sample(data, 0.25)
    assert "isFraud" in s.columns
    assert abs(s.isFraud.mean() - data.isFraud.mean()) < 0.002
