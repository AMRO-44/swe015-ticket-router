"""Load CLINC150 (data_full) + our 100 hand-written appliance examples.

All four models are trained on `train`, tuned on `val` and evaluated on the
SAME held-out `test` split (R3).
"""
import csv
import json
from functools import lru_cache

from .config import RAW_DIR, DATA_DIR, OOS_LABEL, DOMAIN_TO_TEAM, APPLIANCE_INTENT_TO_TEAM, HUMAN_AGENT


@lru_cache(maxsize=1)
def intent_to_domain():
    domains = json.load(open(RAW_DIR / "domains.json", encoding="utf-8"))
    m = {intent: dom for dom, intents in domains.items() for intent in intents}
    for intent in APPLIANCE_INTENT_TO_TEAM:
        m[intent] = "appliance"
    m[OOS_LABEL] = "oos"
    return m


def intent_to_team(intent: str) -> str:
    if intent == OOS_LABEL:
        return HUMAN_AGENT
    if intent in APPLIANCE_INTENT_TO_TEAM:
        return APPLIANCE_INTENT_TO_TEAM[intent]
    return DOMAIN_TO_TEAM.get(intent_to_domain().get(intent, ""), HUMAN_AGENT)


def _appliance_rows():
    rows = {"train": [], "val": [], "test": []}
    with open(DATA_DIR / "appliance_intents.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["split"]].append((r["text"], r["intent"]))
    return rows


@lru_cache(maxsize=1)
def load_splits():
    """Return dict split -> list of (text, intent). OOS examples use label 'oos'."""
    d = json.load(open(RAW_DIR / "data_full.json", encoding="utf-8"))
    app = _appliance_rows()
    splits = {}
    for split, oos_key in [("train", "oos_train"), ("val", "oos_val"), ("test", "oos_test")]:
        rows = [(t, y) for t, y in d[split]] + [(t, OOS_LABEL) for t, _ in d[oos_key]] + app[split]
        splits[split] = rows
    return splits


@lru_cache(maxsize=1)
def label_list():
    return sorted({y for _, y in load_splits()["train"]})


def dataset_summary():
    s = load_splits()
    out = {}
    for split, rows in s.items():
        out[split] = {
            "total": len(rows),
            "in_scope": sum(y != OOS_LABEL for _, y in rows),
            "oos": sum(y == OOS_LABEL for _, y in rows),
            "appliance": sum(y.startswith("appliance_") for _, y in rows),
        }
    out["num_intents"] = len(label_list())
    return out
