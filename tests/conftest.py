import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudnet.config import ModelConfig  # noqa: E402
from fraudnet.data.loader import time_split  # noqa: E402
from fraudnet.data.synthetic import make_synthetic  # noqa: E402
from fraudnet.models.pipeline import FraudPipeline  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return make_synthetic(n=8000, seed=7)


@pytest.fixture(scope="session")
def split(data):
    return time_split(data, 0.25)


@pytest.fixture(scope="session")
def trained(split, tmp_path_factory):
    train, valid = split
    pipe = FraudPipeline(ModelConfig(n_estimators=150), oof_folds=3)
    pipe.fit(train, valid)
    path = tmp_path_factory.mktemp("model") / "fraud_pipeline.joblib"
    pipe.save(path)
    return pipe, path
