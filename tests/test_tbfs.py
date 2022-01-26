import pandas as pd
from tbfs.ranker import TBFSRanker, metrics


def test_tbfs_threshold_count():
    tbfs = TBFSRanker(t_delta=0.05)
    assert tbfs.threshold_count == 20

    tbfs = TBFSRanker(t_delta=0.01)
    assert tbfs.threshold_count == 100

    tbfs = TBFSRanker(t_delta=0.001)
    assert tbfs.threshold_count == 1_000


def test_tbfs_fit_runs():
    x = pd.DataFrame({"a": [1, 5, 2, 6, 0], "b": [1, 2, 3, 4, 5]})
    y = [0, 1, 0, 1, 0]
    tbfs = TBFSRanker(0.01)
    rankings = tbfs.fit(x, y)
    assert rankings is not None


def test_metric_count():
    assert len(metrics) == 12
