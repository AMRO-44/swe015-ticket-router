"""Download the trained weights from the Hugging Face Hub into models/ (R7: weights are not in git).

Usage:  python scripts/download_models.py            (uses HF_REPO_ID from src/config.py or env)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import HF_REPO_ID, MODELS_DIR  # noqa: E402

REQUIRED = ["meta.json", "vocab.json", "tfidf_lr.joblib", "word2vec.npz", "ffn.npz",
            "student_int8.pt", "student/config.json", "teacher/config.json"]


def missing_files():
    return [f for f in REQUIRED if not (MODELS_DIR / f).exists()]


def ensure_models(verbose=True):
    miss = missing_files()
    if not miss:
        return True
    if HF_REPO_ID.startswith("YOUR_HF_USERNAME"):
        if verbose:
            print(f"[download] missing {miss} and HF_REPO_ID is not set - train with "
                  "`python train_all.py` or set HF_REPO_ID.")
        return False
    try:
        from huggingface_hub import snapshot_download
        print(f"[download] fetching weights from https://huggingface.co/{HF_REPO_ID} ...")
        snapshot_download(repo_id=HF_REPO_ID, local_dir=str(MODELS_DIR), ignore_patterns=["teacher_int8.pt", "*.md"])
    except Exception as e:  # network / auth problems should not crash the app
        print(f"[download] failed: {e}")
        return False
    miss = missing_files()
    if miss and verbose:
        print("[download] still missing:", miss)
    return not miss


if __name__ == "__main__":
    ok = ensure_models()
    print("all models present" if ok else "some models are missing")
