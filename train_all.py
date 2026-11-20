"""Train and evaluate ALL models on the same held-out test split, write results/results.json,
then regenerate the architecture diagram, results table/figure and the presentation.

    python train_all.py                    # everything (GPU recommended for the Transformers)
    python train_all.py --skip-transformers  # only classical baseline + word2vec/FFN
"""
import argparse
import subprocess
import sys

from src import train_classical

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformers", action="store_true")
    ap.add_argument("--retrain-teacher", action="store_true")
    a = ap.parse_args()
    train_classical.main()
    if not a.skip_transformers:
        from src import train_transformer
        train_transformer.main(retrain_teacher=a.retrain_teacher)
    for script in ["scripts/make_figures.py", "scripts/make_slides.py"]:
        subprocess.run([sys.executable, script], check=False)
