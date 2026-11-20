"""Train + evaluate the classical baseline (tf-idf + LR) and the neural network (word2vec + FFN)."""
import json

import numpy as np

from .baseline import TfidfLR
from .benchmark import hardware_string, latency_ms, size_mb
from .config import (MODELS_DIR, RAW_DIR, SEED, W2V_DIM, W2V_WINDOW, W2V_NEG, W2V_EPOCHS,
                     FFN_HIDDEN, FFN_EPOCHS, FFN_LR, FFN_MOMENTUM, FFN_DROPOUT)
from .data import load_splits, label_list, dataset_summary
from .evaluate import evaluate_model
from .ffn_np import FFN
from .preprocess import Normalizer, build_vocab, tokenize, basic_normalize
from .results_io import save_model_result
from .word2vec_np import Word2Vec


def get_normalizer(fit=False) -> Normalizer:
    path = MODELS_DIR / "vocab.json"
    if fit or not path.exists():
        vocab = build_vocab([t for t, _ in load_splits()["train"]])
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(sorted(vocab), open(path, "w"))
    return Normalizer(set(json.load(open(path))))


def normalized_splits(norm: Normalizer):
    s = load_splits()
    return {k: ([norm.normalize(t) for t, _ in v], [y for _, y in v]) for k, v in s.items()}


def _errors(texts, y_true, y_pred, k=30):
    return [{"text": t, "true": a, "pred": b} for t, a, b in zip(texts, y_true, y_pred) if a != b][:k]


def run_baseline(norm, S, labels):
    print("\n[1/2] tf-idf + logistic regression baseline")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S["train"], S["val"], S["test"]
    model = TfidfLR().fit(Xtr, ytr, labels)
    m, pred = evaluate_model("TF-IDF + LogReg", model.predict_proba(Xva), yva, model.predict_proba(Xte), yte, labels)
    path = MODELS_DIR / "tfidf_lr.joblib"
    model.save(path)
    m.update(latency_ms(lambda t: model.predict_proba([norm.normalize(t)])))
    m.update(size_mb=size_mb(path), role="Classical baseline", precision="FP64 (sparse)",
             errors=_errors(load_raw_test(), yte, pred))
    print("  ", {k: v for k, v in m.items() if k != "errors"})
    save_model_result("tfidf_lr", m, {"hardware": hardware_string(), "data": dataset_summary(),
                                      "labels": labels})


def load_raw_test():
    return [t for t, _ in load_splits()["test"]]


def run_ffn(norm, S, labels):
    print("\n[2/2] own word2vec (NumPy SGNS) + feedforward network (NumPy, SGD + backprop)")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S["train"], S["val"], S["test"]
    wiki = [basic_normalize(l) for l in open(RAW_DIR / "all_wiki_sents.txt", encoding="utf-8")]
    corpus = [tokenize(t) for t in Xtr + wiki]
    w2v = Word2Vec(dim=W2V_DIM, window=W2V_WINDOW, neg=W2V_NEG, seed=SEED).train(corpus, epochs=W2V_EPOCHS)
    for w in ["dishwasher", "balance", "flight", "weather"]:
        print(f"   nearest({w}) =", [x for x, _ in w2v.most_similar(w, 4)])
    w2v_path = MODELS_DIR / "word2vec.npz"
    w2v.save(w2v_path)

    lab2id = {l: i for i, l in enumerate(labels)}
    feats = lambda texts: np.stack([w2v.sentence_vector(tokenize(t)) for t in texts]).astype(np.float32)
    Ftr, Fva, Fte = feats(Xtr), feats(Xva), feats(Xte)
    ytr_id, yva_id = np.array([lab2id[y] for y in ytr]), np.array([lab2id[y] for y in yva])
    ffn = FFN(W2V_DIM, FFN_HIDDEN, len(labels), dropout=FFN_DROPOUT, seed=SEED)
    hist = ffn.fit(Ftr, ytr_id, Fva, yva_id, epochs=FFN_EPOCHS, lr=FFN_LR, momentum=FFN_MOMENTUM)
    ffn_path = MODELS_DIR / "ffn.npz"
    ffn.save(ffn_path)
    json.dump(hist, open(MODELS_DIR / "ffn_history.json", "w"))

    m, pred = evaluate_model("word2vec + FFN", ffn.predict_proba(Fva), yva, ffn.predict_proba(Fte), yte, labels)
    m.update(latency_ms(lambda t: ffn.predict_proba(feats([norm.normalize(t)]))))
    m.update(size_mb=size_mb(w2v_path, ffn_path), role="Neural network (own)", precision="FP32",
             errors=_errors(load_raw_test(), yte, pred), w2v_vocab=len(w2v.vocab))
    print("  ", {k: v for k, v in m.items() if k != "errors"})
    save_model_result("w2v_ffn", m)


def main():
    np.random.seed(SEED)
    norm = get_normalizer(fit=True)
    S = normalized_splits(norm)
    labels = label_list()
    run_baseline(norm, S, labels)
    run_ffn(norm, S, labels)


if __name__ == "__main__":
    main()
