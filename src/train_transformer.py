"""Train + evaluate the Transformer rungs of the ladder:

  bert_base       BERT-base-uncased fine-tuned (FP32)            <- "Transformer"
  bert_base_int8  same model, dynamic INT8                        (extra row)
  student_fp32    BERT-small distilled from the teacher (FP32)    (extra row)
  student_int8    distilled student, dynamic INT8                 <- "quantized Transformer", used by the app

Accuracy is computed on the same normalized test split as the other models;
latency is ALWAYS measured on CPU with one message at a time.
"""
import argparse
import json

import numpy as np
import torch

from .benchmark import hardware_string, latency_ms, size_mb
from .config import (MODELS_DIR, TEACHER_NAME, STUDENT_NAME, TEACHER_EPOCHS, TEACHER_LR,
                     STUDENT_EPOCHS, STUDENT_LR, DISTILL_T, DISTILL_ALPHA)
from .data import label_list, load_splits
from .evaluate import evaluate_model
from .results_io import save_model_result
from .train_classical import get_normalizer, normalized_splits, _errors, load_raw_test
from . import transformer as tr

TEACHER_DIR = MODELS_DIR / "teacher"
STUDENT_DIR = MODELS_DIR / "student"


def _finish(model_id, name, role, precision, cpu_model, tok, norm, probs_va, yva, probs_te, yte, labels, size, extra=None):
    m, pred = evaluate_model(name, probs_va, yva, probs_te, yte, labels)
    m.update(latency_ms(lambda t: tr.predict_proba(cpu_model, tok, [norm.normalize(t)], device=torch.device("cpu"))))
    m.update(size_mb=size, role=role, precision=precision, errors=_errors(load_raw_test(), yte, pred))
    if extra:
        m.update(extra)
    print("  ", {k: v for k, v in m.items() if k != "errors"})
    save_model_result(model_id, m, {"hardware": hardware_string()})
    return m


def main(retrain_teacher=False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    labels = label_list()
    lab2id = {l: i for i, l in enumerate(labels)}
    norm = get_normalizer()
    S = normalized_splits(norm)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S["train"], S["val"], S["test"]
    ytr_id, yva_id = [lab2id[y] for y in ytr], [lab2id[y] for y in yva]
    dev = tr.get_device()
    cpu = torch.device("cpu")
    print(f"device for training: {dev}")

    # ---------------- 1. teacher: BERT-base fine-tuned ----------------
    if retrain_teacher or not (TEACHER_DIR / "config.json").exists():
        print("\n[1/4] fine-tuning teacher", TEACHER_NAME)
        teacher, tok, hist = tr.train_teacher(TEACHER_NAME, labels, Xtr, ytr_id, Xva, yva_id, TEACHER_EPOCHS, TEACHER_LR)
        teacher.save_pretrained(TEACHER_DIR)
        tok.save_pretrained(TEACHER_DIR)
        json.dump(hist, open(TEACHER_DIR / "history.json", "w"), indent=2)
    else:
        print("\n[1/4] re-using saved teacher in", TEACHER_DIR)
        teacher, tok = tr.load_fp32(TEACHER_DIR), tr.load_tokenizer(TEACHER_DIR)
    teacher.to(dev)
    pva, pte = tr.predict_proba(teacher, tok, Xva), tr.predict_proba(teacher, tok, Xte)
    teacher_train_logits = tr.predict_logits(teacher, tok, Xtr)
    teacher_cpu = teacher.to(cpu).eval()
    _finish("bert_base", "BERT-base fine-tuned", "Transformer", "FP32", teacher_cpu, tok, norm,
            pva, yva, pte, yte, labels, size_mb(TEACHER_DIR / "model.safetensors", TEACHER_DIR / "pytorch_model.bin"),
            {"params_m": round(sum(p.numel() for p in teacher.parameters()) / 1e6, 1)})

    # ---------------- 2. teacher INT8 (extra row) ----------------
    print("\n[2/4] dynamic INT8 quantization of the teacher")
    teacher_q = tr.quantize_dynamic_int8(teacher_cpu)
    tr.save_state(teacher_q, MODELS_DIR / "teacher_int8.pt")
    pva, pte = tr.predict_proba(teacher_q, tok, Xva, device=cpu), tr.predict_proba(teacher_q, tok, Xte, device=cpu)
    _finish("bert_base_int8", "BERT-base INT8", "Transformer, quantized (extra)", "INT8 dynamic", teacher_q, tok,
            norm, pva, yva, pte, yte, labels, size_mb(MODELS_DIR / "teacher_int8.pt"))
    del teacher_q

    # ---------------- 3. distillation into BERT-small ----------------
    print("\n[3/4] knowledge distillation into", STUDENT_NAME)
    student, hist = tr.train_student(STUDENT_NAME, labels, tok, Xtr, ytr_id, teacher_train_logits, Xva, yva_id,
                                     STUDENT_EPOCHS, STUDENT_LR, DISTILL_T, DISTILL_ALPHA)
    student.save_pretrained(STUDENT_DIR)
    tok.save_pretrained(STUDENT_DIR)
    json.dump(hist, open(STUDENT_DIR / "history.json", "w"), indent=2)
    pva, pte = tr.predict_proba(student, tok, Xva), tr.predict_proba(student, tok, Xte)
    student_cpu = student.to(cpu).eval()
    _finish("student_fp32", "Distilled BERT-small FP32", "Distilled Transformer (extra)", "FP32", student_cpu, tok,
            norm, pva, yva, pte, yte, labels,
            size_mb(STUDENT_DIR / "model.safetensors", STUDENT_DIR / "pytorch_model.bin"),
            {"params_m": round(sum(p.numel() for p in student.parameters()) / 1e6, 1)})

    # ---------------- 4. student INT8 = model used by the app ----------------
    print("\n[4/4] dynamic INT8 quantization of the student")
    student_q = tr.quantize_dynamic_int8(student_cpu)
    tr.save_state(student_q, MODELS_DIR / "student_int8.pt")
    # sanity check: the saved file reloads into an identical model (this is what the app does)
    reloaded = tr.load_quantized(STUDENT_DIR, MODELS_DIR / "student_int8.pt")
    pva, pte = tr.predict_proba(reloaded, tok, Xva, device=cpu), tr.predict_proba(reloaded, tok, Xte, device=cpu)
    _finish("student_int8", "Distilled BERT-small INT8", "Quantized Transformer (app)", "INT8 dynamic", reloaded,
            tok, norm, pva, yva, pte, yte, labels, size_mb(MODELS_DIR / "student_int8.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain-teacher", action="store_true")
    main(ap.parse_args().retrain_teacher)
