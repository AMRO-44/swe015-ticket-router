"""Area A - feedforward neural network in pure NumPy, trained with mini-batch SGD
(momentum) and hand-written backpropagation.

Architecture:  x (averaged word2vec, D) -> Linear(D,H) -> ReLU -> Dropout(p) -> Linear(H,C) -> softmax
Loss: cross-entropy + L2 weight decay.
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class FFN:
    def __init__(self, d_in, d_hidden, n_classes, dropout=0.3, seed=42):
        self.rng = np.random.default_rng(seed)
        # He initialization for the ReLU layer
        self.params = {
            "W1": (self.rng.standard_normal((d_in, d_hidden)) * np.sqrt(2 / d_in)).astype(np.float32),
            "b1": np.zeros(d_hidden, dtype=np.float32),
            "W2": (self.rng.standard_normal((d_hidden, n_classes)) * np.sqrt(1 / d_hidden)).astype(np.float32),
            "b2": np.zeros(n_classes, dtype=np.float32),
        }
        self.p = dropout
        self.mu = None   # input standardization (fitted on train)
        self.sd = None

    # ---------- forward ----------
    def forward(self, X, train=False):
        P = self.params
        z1 = X @ P["W1"] + P["b1"]
        h = np.maximum(z1, 0)
        mask = None
        if train and self.p > 0:
            mask = (self.rng.random(h.shape) > self.p).astype(np.float32) / (1 - self.p)  # inverted dropout
            h = h * mask
        logits = h @ P["W2"] + P["b2"]
        return logits, (X, z1, h, mask)

    # ---------- backward ----------
    def backward(self, probs, y, cache, wd):
        X, z1, h, mask = cache
        P, n = self.params, len(y)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1          # d CE / d logits = softmax - onehot
        dlogits /= n
        g = {"W2": h.T @ dlogits + wd * P["W2"], "b2": dlogits.sum(0)}
        dh = dlogits @ P["W2"].T
        if mask is not None:
            dh *= mask
        dz1 = dh * (z1 > 0)                    # ReLU derivative
        g["W1"] = X.T @ dz1 + wd * P["W1"]
        g["b1"] = dz1.sum(0)
        return g

    def fit(self, X, y, Xval=None, yval=None, epochs=60, lr=0.05, momentum=0.9, batch=64, wd=1e-4, verbose=True):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        vel = {k: np.zeros_like(v) for k, v in self.params.items()}
        history = []
        for ep in range(epochs):
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))     # cosine schedule
            perm = self.rng.permutation(len(X))
            loss_sum = 0.0
            for b in range(0, len(X), batch):
                idx = perm[b:b + batch]
                logits, cache = self.forward(X[idx], train=True)
                probs = softmax(logits)
                loss_sum += -np.log(probs[np.arange(len(idx)), y[idx]] + 1e-9).sum()
                grads = self.backward(probs, y[idx], cache, wd)
                for k in self.params:                                # SGD with momentum
                    vel[k] = momentum * vel[k] - lr_ep * grads[k]
                    self.params[k] += vel[k]
            rec = {"epoch": ep + 1, "train_loss": float(loss_sum / len(X))}
            if Xval is not None:
                rec["val_acc"] = float((self.predict_proba(Xval).argmax(1) == yval).mean())
            history.append(rec)
            if verbose and (ep % 10 == 9 or ep == 0):
                print("  ffn", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        return history

    def predict_proba(self, X):
        X = (X - self.mu) / self.sd
        return softmax(self.forward(X.astype(np.float32))[0])

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, sd=self.sd, p=self.p, **self.params)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z["W1"].shape[0], z["W1"].shape[1], z["W2"].shape[1], float(z["p"]))
        for k in ["W1", "b1", "W2", "b2"]:
            m.params[k] = z[k]
        m.mu, m.sd = z["mu"], z["sd"]
        return m
