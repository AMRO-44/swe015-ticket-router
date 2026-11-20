"""Area B - classical baseline: tf-idf (word 1-2 grams + char 2-4 grams) + logistic regression."""
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import SEED


class TfidfLR:
    def __init__(self, C: float = 10.0):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
        self.labels = None

    def _feats(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, y, labels):
        self.labels = list(labels)
        lab2id = {l: i for i, l in enumerate(self.labels)}
        self.clf.fit(self._feats(texts, fit=True), np.array([lab2id[t] for t in y]))
        return self

    def predict_proba(self, texts) -> np.ndarray:
        p = self.clf.predict_proba(self._feats(texts))
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        out[:, self.clf.classes_] = p
        return out

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)
