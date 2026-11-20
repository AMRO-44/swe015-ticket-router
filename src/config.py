"""Central configuration: paths, model names, hyper-parameters, routing table."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

# Hugging Face Hub repo that stores the trained weights (R7: no weights in git).
# Change this to your own repo, e.g. "my-hf-username/swe015-ticket-router".
HF_REPO_ID = os.environ.get("HF_REPO_ID", "YOUR_HF_USERNAME/swe015-ticket-router")

SEED = 42
OOS_LABEL = "oos"

# ---------------- Transformer settings (Area C / D) ----------------
TEACHER_NAME = "google-bert/bert-base-uncased"                # 110M params, fine-tuned teacher
STUDENT_NAME = "google/bert_uncased_L-4_H-512_A-8"  # BERT-small, 4 layers, 29M params
MAX_LEN = 64
TEACHER_EPOCHS = 4
TEACHER_LR = 4e-5
STUDENT_EPOCHS = 10
STUDENT_LR = 1e-4
BATCH_SIZE = 32
DISTILL_T = 2.0      # softmax temperature
DISTILL_ALPHA = 0.5  # weight of the hard-label CE loss

# ---------------- Word2vec + FFN (Area A) ----------------
W2V_DIM = 100
W2V_WINDOW = 5
W2V_NEG = 5
W2V_EPOCHS = 15
FFN_HIDDEN = 512
FFN_EPOCHS = 60
FFN_LR = 0.05
FFN_MOMENTUM = 0.9
FFN_DROPOUT = 0.3

# ---------------- Benchmark (R4) ----------------
LATENCY_RUNS = 200   # >= 100 required
LATENCY_WARMUP = 10

# ---------------- Routing table ----------------
# CLINC150 domains are used as proxy support teams; the hand-written appliance
# domain is split into three appliance teams.
DOMAIN_TO_TEAM = {
    "banking": "Banking & Payments team",
    "credit_cards": "Credit Cards team",
    "kitchen_and_dining": "Kitchen & Dining team",
    "home": "Smart Home team",
    "auto_and_commute": "Auto & Commute team",
    "travel": "Travel team",
    "utility": "Utility & Info team",
    "work": "Work & HR team",
    "small_talk": "Chatbot (ELIZA small-talk reply)",
    "meta": "Assistant settings team",
}
APPLIANCE_INTENT_TO_TEAM = {
    "appliance_water_problem": "Appliance Technical Service team",
    "appliance_wont_start": "Appliance Technical Service team",
    "appliance_warranty_repair": "Warranty & Repair team",
    "appliance_spare_parts": "Spare Parts & Orders team",
    "appliance_installation": "Installation team",
}
HUMAN_AGENT = "Human agent (low confidence / out-of-scope)"
