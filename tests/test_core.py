"""Unit tests for the classical / NumPy parts:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eliza import eliza_reply  # noqa: E402
from src.evaluate import metrics, predict_with_threshold  # noqa: E402
from src.ffn_np import FFN, softmax  # noqa: E402
from src.preprocess import Normalizer, edit_distance, extract_model_codes, basic_normalize  # noqa: E402


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("dishwaser", "dishwasher") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_model_codes_and_normalization():
    t = "Bulaşık makinem su BOŞALTMIYOR!!! model DW-6534"
    assert extract_model_codes(t) == ["DW-6534"]
    assert basic_normalize(t) == "bulasik makinem su bosaltmiyor model modelcode"
    assert basic_normalize("sooooo bad") == "soo bad"


def test_product_correction():
    n = Normalizer(vocab={"my", "is", "leaking"})
    r = n("my dishwaser is leaking")
    assert r.normalized == "my dishwasher is leaking"
    assert r.corrections == [("dishwaser", "dishwasher", 1)]


def test_eliza():
    assert eliza_reply("hello there") is not None
    assert eliza_reply("my oven is broken") is None
    assert "upset" in eliza_reply("i am upset")


def test_threshold_and_metrics():
    labels = ["a", "b", "oos"]
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.1, 0.8, 0.1]])
    pred = predict_with_threshold(probs, labels, 0.5)
    assert pred == ["a", "oos", "b"]
    m = metrics(["a", "oos", "a"], pred)
    assert m["in_scope_acc"] == 0.5 and m["oos_recall"] == 1.0


def test_ffn_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    net = FFN(5, 7, 3, dropout=0.0, seed=0)
    for k in net.params:
        net.params[k] = net.params[k].astype(np.float64) + (0.1 * rng.standard_normal(net.params[k].shape) if k.startswith("b") else 0)
    X, y = rng.standard_normal((4, 5)), np.array([0, 2, 1, 2])

    def loss():
        p = softmax(net.forward(X)[0])
        return -np.log(p[np.arange(4), y]).mean()

    logits, cache = net.forward(X, train=True)
    g = net.backward(softmax(logits), y, cache, wd=0.0)
    for k in ["W1", "b1", "W2", "b2"]:
        P = net.params[k]
        idx = tuple(rng.integers(0, s) for s in P.shape)
        old = P[idx]
        P[idx] = old + 1e-5; lp = loss()
        P[idx] = old - 1e-5; lm = loss()
        P[idx] = old
        assert abs((lp - lm) / 2e-5 - g[k][idx]) < 1e-6, k
