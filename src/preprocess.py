"""Area B - classical NLP preprocessing.

* regex text normalization (Turkish characters, URLs, e-mails, repeated letters, punctuation)
* regex extraction of appliance model codes such as "DW-6534"
* minimum-edit-distance (Levenshtein) correction of misspelled product names
"""
import re
from collections import Counter
from dataclasses import dataclass, field

# Turkish-aware lower-casing + ASCII folding (bulaşık -> bulasik)
_TR_MAP = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
                         "â": "a", "î": "i", "û": "u"})
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{2,3})[-]?(\d{3,5})([A-Za-z]?)\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT_RE = re.compile(r"(\w)\1{2,}")          # "sooooo" -> "soo"
NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9']+")

MODEL_TOKEN = "modelcode"

# Product / domain lexicon used as correction targets (English + ASCII-folded Turkish)
PRODUCT_LEXICON = [
    "dishwasher", "washer", "washing", "machine", "refrigerator", "fridge", "freezer",
    "oven", "microwave", "dryer", "cooktop", "stove", "warranty", "installation",
    "technician", "filter", "appliance", "drain", "leaking",
    "bulasik", "makinesi", "makinem", "camasir", "buzdolabi", "firin", "kurutma",
    "garanti", "kurulum", "servis", "yedek", "parca",
]


def edit_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Levenshtein distance with unit costs (insert / delete / substitute), O(len(a)*len(b)) DP."""
    if a == b:
        return 0
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,              # deletion
                         cur[j - 1] + 1,           # insertion
                         prev[j - 1] + (ca != cb))  # substitution
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def fold(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return text.translate(_TR_MAP)


def extract_model_codes(text: str) -> list[str]:
    return [f"{a.upper()}-{n}{s.upper()}" for a, n, s in MODEL_CODE_RE.findall(text)]


def basic_normalize(text: str) -> str:
    """Normalization without spelling correction (used to build the vocabulary)."""
    text = MODEL_CODE_RE.sub(f" {MODEL_TOKEN} ", text)
    text = URL_RE.sub(" url ", text)
    text = EMAIL_RE.sub(" email ", text)
    text = fold(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_vocab(texts, min_count: int = 1) -> set[str]:
    c = Counter(tok for t in texts for tok in tokenize(basic_normalize(t)))
    return {w for w, n in c.items() if n >= min_count}


@dataclass
class NormResult:
    original: str
    normalized: str
    model_codes: list = field(default_factory=list)
    corrections: list = field(default_factory=list)   # (wrong, corrected, distance)


class Normalizer:
    """Full Area-B preprocessing pipeline applied to EVERY model's input."""

    def __init__(self, vocab: set[str], lexicon=PRODUCT_LEXICON):
        self.vocab = vocab
        self.lexicon = list(lexicon)

    def correct_token(self, tok: str):
        if tok in self.vocab or tok in self.lexicon or len(tok) < 4 or tok.isdigit():
            return tok, None
        max_d = 1 if len(tok) <= 5 else 2
        best, best_d = None, max_d + 1
        for w in self.lexicon:
            d = edit_distance(tok, w, max_d)
            if d < best_d:
                best, best_d = w, d
        if best is not None and best_d <= max_d:
            return best, best_d
        return tok, None

    def __call__(self, text: str) -> NormResult:
        codes = extract_model_codes(text)
        toks = tokenize(basic_normalize(text))
        out, corr = [], []
        for t in toks:
            c, d = self.correct_token(t)
            if d is not None:
                corr.append((t, c, d))
            out.append(c)
        return NormResult(text, " ".join(out), codes, corr)

    def normalize(self, text: str) -> str:
        return self(text).normalized
