"""Read / merge results/results.json (every number in the slides comes from this file)."""
import json
import time

from .config import RESULTS_DIR, MODELS_DIR

RESULTS_FILE = RESULTS_DIR / "results.json"

# ladder order used in tables / slides
MODEL_ORDER = ["tfidf_lr", "w2v_ffn", "bert_base", "bert_base_int8", "student_fp32", "student_int8"]


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.load(open(RESULTS_FILE, encoding="utf-8"))
    return {"models": {}}


def save_model_result(model_id: str, entry: dict, extra: dict | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r = load_results()
    entry["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    r["models"][model_id] = entry
    if extra:
        r.update(extra)
    json.dump(r, open(RESULTS_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    # thresholds are also stored next to the weights so the app can use them
    if "threshold" in entry:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = MODELS_DIR / "meta.json"
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        meta.setdefault("thresholds", {})[model_id] = entry["threshold"]
        json.dump(meta, open(meta_path, "w"), indent=2)
    return r
