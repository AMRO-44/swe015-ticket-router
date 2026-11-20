"""Areas C + D - BERT teacher fine-tuning, knowledge distillation into a small BERT
student, and dynamic INT8 quantization (PyTorch).

Both models use the same uncased WordPiece vocabulary (the BERT miniatures were
trained with the bert-base-uncased vocab), so one tokenizer serves both.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import MAX_LEN, BATCH_SIZE, SEED


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(name_or_dir):
    return AutoTokenizer.from_pretrained(name_or_dir)


def _encode(tok, texts, pad_to_max=True):
    return tok(list(texts), padding="max_length" if pad_to_max else True, truncation=True,
               max_length=MAX_LEN, return_tensors="pt")


def new_classifier(name, labels):
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)})


@torch.no_grad()
def predict_logits(model, tok, texts, batch_size=64, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _encode(tok, texts[i:i + batch_size], pad_to_max=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def predict_proba(model, tok, texts, batch_size=64, device=None):
    logits = predict_logits(model, tok, texts, batch_size, device)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _train_loop(model, tok, texts, y, loss_fn, epochs, lr, val_texts, val_y, extra=None, tag="model"):
    """Generic fine-tuning loop with AdamW + linear warm-up/decay; keeps the best epoch on val."""
    device = get_device()
    model.to(device)
    enc = _encode(tok, texts)
    y_t = torch.tensor(y, dtype=torch.long)
    tensors = [enc["input_ids"], enc["attention_mask"], y_t] + ([torch.tensor(extra)] if extra is not None else [])
    ds = torch.utils.data.TensorDataset(*tensors)
    g = torch.Generator().manual_seed(SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_acc, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for batch in dl:
            batch = [b.to(device) for b in batch]
            ids, mask, yb = batch[:3]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits.float(), yb, batch[3] if extra is not None else None)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(yb)
        val_acc = float((predict_logits(model, tok, val_texts).argmax(1) == np.array(val_y)).mean())
        history.append({"epoch": ep + 1, "train_loss": loss_sum / len(ds), "val_acc": val_acc,
                        "seconds": round(time.time() - t0, 1)})
        print(f"  {tag} epoch {ep + 1}/{epochs} loss={loss_sum / len(ds):.4f} val_acc={val_acc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def train_teacher(name, labels, texts, y, val_texts, val_y, epochs, lr):
    set_seed()
    tok = load_tokenizer(name)
    model = new_classifier(name, labels)
    ce = lambda logits, yb, _: F.cross_entropy(logits, yb)
    model, hist = _train_loop(model, tok, texts, y, ce, epochs, lr, val_texts, val_y, tag="teacher")
    return model, tok, hist


def train_student(name, labels, tok, texts, y, teacher_logits, val_texts, val_y, epochs, lr, T, alpha):
    """Knowledge distillation (Hinton et al., 2015):
       L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))"""
    set_seed()
    model = new_classifier(name, labels)

    def kd_loss(s_logits, yb, t_logits):
        hard = F.cross_entropy(s_logits, yb)
        soft = F.kl_div(F.log_softmax(s_logits / T, dim=-1), F.softmax(t_logits / T, dim=-1),
                        reduction="batchmean") * (T * T)
        return alpha * hard + (1 - alpha) * soft

    model, hist = _train_loop(model, tok, texts, y, kd_loss, epochs, lr, val_texts, val_y,
                              extra=teacher_logits.astype(np.float32), tag="student")
    return model, hist


# ---------------------------------------------------------------- quantization (Area D)
def _pick_engine():
    engines = torch.backends.quantized.supported_engines
    for e in ("fbgemm", "x86", "qnnpack"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def quantize_dynamic_int8(model):
    """Post-training dynamic quantization: nn.Linear weights -> INT8, activations quantized on
    the fly at inference time. Embeddings and LayerNorm stay FP32."""
    _pick_engine()
    model = copy.deepcopy(model).cpu().eval()
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:  # very old torch
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def save_state(model, path):
    torch.save(model.state_dict(), path)


def load_quantized(config_dir, state_path):
    """Rebuild the architecture from its config, quantize it, then load the INT8 weights."""
    config = AutoConfig.from_pretrained(config_dir)
    model = AutoModelForSequenceClassification.from_config(config)
    q = quantize_dynamic_int8(model)
    q.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    return q.eval()


def load_fp32(model_dir):
    return AutoModelForSequenceClassification.from_pretrained(model_dir).cpu().eval()
