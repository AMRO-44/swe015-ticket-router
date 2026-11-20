"""Shared evaluation: every model outputs a probability matrix over the same label list.

A prediction with max-probability below a threshold tau is mapped to 'oos'
(=> human agent). tau is tuned on the validation split only, then applied to test.
Metrics (CLINC150 protocol): in-scope accuracy, out-of-scope recall, overall accuracy.
"""
import numpy as np

from .config import OOS_LABEL


def predict_with_threshold(probs: np.ndarray, labels: list[str], tau: float) -> list[str]:
    idx = probs.argmax(1)
    conf = probs.max(1)
    return [OOS_LABEL if c < tau else labels[i] for i, c in zip(idx, conf)]


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    yt, yp = np.array(y_true), np.array(y_pred)
    ins = yt != OOS_LABEL
    return {
        "in_scope_acc": float((yt[ins] == yp[ins]).mean()),
        "oos_recall": float((yp[~ins] == OOS_LABEL).mean()) if (~ins).any() else float("nan"),
        "overall_acc": float((yt == yp).mean()),
    }


def tune_threshold(probs_val: np.ndarray, y_val: list[str], labels: list[str]) -> float:
    """Pick tau in [0, 0.95] maximizing the mean of in-scope accuracy and OOS recall on val."""
    best_tau, best = 0.0, -1.0
    for tau in np.round(np.arange(0.0, 0.96, 0.01), 2):
        m = metrics(y_val, predict_with_threshold(probs_val, labels, tau))
        score = (m["in_scope_acc"] + m["oos_recall"]) / 2
        if score > best + 1e-9:
            best, best_tau = score, float(tau)
    return best_tau


def evaluate_model(name, probs_val, y_val, probs_test, y_test, labels) -> dict:
    tau = tune_threshold(probs_val, y_val, labels)
    pred = predict_with_threshold(probs_test, labels, tau)
    m = metrics(y_test, pred)
    m["threshold"] = tau
    m["name"] = name
    return m, pred
