"""Area B/A - our own word2vec (skip-gram with negative sampling), written in NumPy.

Trained only on the training utterances + the unlabeled Wikipedia sentences that
ship with CLINC150 (never on val/test).

Loss per (center c, context o) pair with K negatives n_k:
    L = -log s(u_o . v_c) - sum_k log s(-u_nk . v_c)
Gradients (s = sigmoid):
    dL/dv_c  = (s(u_o.v_c) - 1) u_o + sum_k s(u_nk.v_c) u_nk
    dL/du_o  = (s(u_o.v_c) - 1) v_c
    dL/du_nk = s(u_nk.v_c) v_c
"""
from collections import Counter

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class Word2Vec:
    def __init__(self, dim=100, window=3, neg=5, min_count=2, seed=42):
        self.dim, self.window, self.neg, self.min_count = dim, window, neg, min_count
        self.rng = np.random.default_rng(seed)
        self.vocab, self.W = {}, None

    def build(self, sentences):
        c = Counter(w for s in sentences for w in s)
        words = [w for w, n in c.most_common() if n >= self.min_count]
        self.vocab = {w: i for i, w in enumerate(words)}
        counts = np.array([c[w] for w in words], dtype=np.float64)
        self.noise = counts ** 0.75
        self.noise /= self.noise.sum()
        # sub-sampling of frequent words (Mikolov et al. 2013)
        f = counts / counts.sum()
        self.keep_p = np.minimum(1.0, np.sqrt(1e-4 / f) + 1e-4 / f)

    def _pairs(self, sentences):
        centers, contexts = [], []
        for s in sentences:
            ids = [self.vocab[w] for w in s if w in self.vocab]
            ids = [i for i in ids if self.rng.random() < self.keep_p[i]]
            for pos, c in enumerate(ids):
                w = self.rng.integers(1, self.window + 1)   # dynamic window
                for o in ids[max(0, pos - w): pos] + ids[pos + 1: pos + 1 + w]:
                    centers.append(c)
                    contexts.append(o)
        return np.array(centers), np.array(contexts)

    def train(self, sentences, epochs=5, lr=0.025, batch=512, verbose=True):
        if not self.vocab:
            self.build(sentences)
        V, D = len(self.vocab), self.dim
        self.W = (self.rng.random((V, D), dtype=np.float32) - 0.5) / D   # input vectors v
        C = np.zeros((V, D), dtype=np.float32)                            # output vectors u
        for ep in range(epochs):
            cen, ctx = self._pairs(sentences)
            perm = self.rng.permutation(len(cen))
            cen, ctx = cen[perm], ctx[perm]
            total, loss_sum = len(cen), 0.0
            for b in range(0, total, batch):
                alpha = lr * max(0.0001, 1 - (ep * total + b) / (epochs * total))  # linear decay
                c, o = cen[b:b + batch], ctx[b:b + batch]
                n = self.rng.choice(V, size=(len(c), self.neg), p=self.noise)
                vc = self.W[c]                                   # (B, D)
                uo = C[o]                                        # (B, D)
                un = C[n]                                        # (B, K, D)
                s_pos = _sigmoid((vc * uo).sum(1))               # (B,)
                s_neg = _sigmoid(np.einsum("bkd,bd->bk", un, vc))  # (B, K)
                loss_sum += -(np.log(s_pos + 1e-9).sum() + np.log(1 - s_neg + 1e-9).sum())
                g_pos = (s_pos - 1)[:, None]                     # (B, 1)
                grad_vc = g_pos * uo + np.einsum("bk,bkd->bd", s_neg, un)
                grad_uo = g_pos * vc
                grad_un = s_neg[:, :, None] * vc[:, None, :]
                np.add.at(self.W, c, -alpha * grad_vc)
                np.add.at(C, o, -alpha * grad_uo)
                np.add.at(C, n.ravel(), -alpha * grad_un.reshape(-1, D))
            if verbose:
                print(f"  word2vec epoch {ep + 1}/{epochs}  pairs={total}  loss/pair={loss_sum / total:.4f}")
        return self

    def most_similar(self, word, k=5):
        if word not in self.vocab:
            return []
        Wn = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-9)
        sims = Wn @ Wn[self.vocab[word]]
        inv = {i: w for w, i in self.vocab.items()}
        return [(inv[i], float(sims[i])) for i in np.argsort(-sims)[1:k + 1]]

    def sentence_vector(self, tokens):
        ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        return self.W[ids].mean(0)

    def save(self, path):
        words = np.array(sorted(self.vocab, key=self.vocab.get))
        np.savez_compressed(path, W=self.W.astype(np.float32), words=words, dim=self.dim)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls(dim=int(z["dim"]))
        m.W = z["W"]
        m.vocab = {w: i for i, w in enumerate(z["words"].tolist())}
        return m
