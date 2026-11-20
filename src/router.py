"""Inference pipeline used by the app (see docs/architecture.png).

message -> Normalizer (regex, model codes, edit-distance product correction)
        -> ELIZA small-talk rules
        -> [TF-IDF+LR | word2vec+FFN | BERT-base FP32 | distilled BERT-small INT8]
        -> confidence threshold (tau tuned on val) -> team / human agent

The routing DECISION is made by the quantized Transformer (student_int8, R5);
the other models are shown side by side for comparison.
"""
import json
import time

import numpy as np

from .config import MODELS_DIR, OOS_LABEL, HUMAN_AGENT
from .data import intent_to_domain, intent_to_team, label_list
from .eliza import eliza_reply
from .preprocess import tokenize
from .train_classical import get_normalizer

DISPLAY = {
    "tfidf_lr": "TF-IDF + LogReg (classical)",
    "w2v_ffn": "word2vec + FFN (NumPy)",
    "bert_base": "BERT-base FP32 (Transformer)",
    "student_int8": "Distilled BERT-small INT8 (quantized)",
}
PRIMARY = "student_int8"


class Router:
    def __init__(self, models_dir=MODELS_DIR, load_teacher=True):
        self.models_dir = models_dir
        self.labels = label_list()
        self.norm = get_normalizer()
        meta_path = models_dir / "meta.json"
        self.tau = json.load(open(meta_path)).get("thresholds", {}) if meta_path.exists() else {}
        self.models = {}
        self._load_classical()
        self._load_transformers(load_teacher)
        self.primary = PRIMARY if PRIMARY in self.models else (next(iter(self.models)) if self.models else None)

    # ------------------------------------------------------------------ loading
    def _load_classical(self):
        p = self.models_dir / "tfidf_lr.joblib"
        if p.exists():
            from .baseline import TfidfLR
            m = TfidfLR.load(p)
            self.models["tfidf_lr"] = lambda texts, m=m: m.predict_proba(texts)
        if (self.models_dir / "ffn.npz").exists() and (self.models_dir / "word2vec.npz").exists():
            from .ffn_np import FFN
            from .word2vec_np import Word2Vec
            ffn, w2v = FFN.load(self.models_dir / "ffn.npz"), Word2Vec.load(self.models_dir / "word2vec.npz")
            self.models["w2v_ffn"] = lambda texts: ffn.predict_proba(
                np.stack([w2v.sentence_vector(tokenize(t)) for t in texts]).astype(np.float32))

    def _load_transformers(self, load_teacher):
        student_ok = (self.models_dir / "student_int8.pt").exists() and (self.models_dir / "student" / "config.json").exists()
        teacher_ok = load_teacher and (self.models_dir / "teacher" / "config.json").exists()
        if not (student_ok or teacher_ok):
            return
        import torch
        from . import transformer as tr
        cpu = torch.device("cpu")
        if teacher_ok:
            t_model, t_tok = tr.load_fp32(self.models_dir / "teacher"), tr.load_tokenizer(self.models_dir / "teacher")
            self.models["bert_base"] = lambda texts: tr.predict_proba(t_model, t_tok, texts, device=cpu)
        if student_ok:
            s_model = tr.load_quantized(self.models_dir / "student", self.models_dir / "student_int8.pt")
            s_tok = tr.load_tokenizer(self.models_dir / "student")
            self.models["student_int8"] = lambda texts: tr.predict_proba(s_model, s_tok, texts, device=cpu)

    # ------------------------------------------------------------------ inference
    def _predict(self, model_id, normalized):
        t0 = time.perf_counter()
        probs = self.models[model_id]([normalized])[0]
        ms = (time.perf_counter() - t0) * 1000
        i = int(np.argmax(probs))
        conf = float(probs[i])
        tau = float(self.tau.get(model_id, 0.5))
        intent = self.labels[i] if conf >= tau else OOS_LABEL
        return {"model": DISPLAY.get(model_id, model_id), "top_intent": self.labels[i], "confidence": round(conf, 3),
                "threshold": tau, "decision": intent, "team": intent_to_team(intent), "latency_ms": round(ms, 2)}

    def route(self, text: str) -> dict:
        text = (text or "").strip()
        nr = self.norm(text)
        rows = [self._predict(mid, nr.normalized) for mid in DISPLAY if mid in self.models]
        primary = next((r for r in rows if r["model"] == DISPLAY.get(self.primary)), None)

        reply = None
        rule = eliza_reply(text) if len(tokenize(nr.normalized)) <= 8 else None
        if rule:                                             # rule-based small talk wins
            team, reply, source = "Chatbot (ELIZA small-talk reply)", rule, "ELIZA regex rule"
        elif primary is None:
            team, source = HUMAN_AGENT, "no model loaded"
        else:
            team, source = primary["team"], primary["model"]
            if intent_to_domain().get(primary["decision"]) == "small_talk":
                reply = eliza_reply(text, force=True)
        return {"normalized": nr.normalized, "model_codes": nr.model_codes, "corrections": nr.corrections,
                "team": team, "decided_by": source, "reply": reply, "rows": rows,
                "intent": primary["decision"] if primary else None,
                "confidence": primary["confidence"] if primary else None}
