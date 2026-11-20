"""Build the 12-slide presentation (docs/presentation.pptx, + PDF if LibreOffice is installed).

Every number is read from results/results.json (produced by train_all.py), so the
slides always match your own runs. Personal info comes from slides_info.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from src.config import (DOCS_DIR, RESULTS_DIR, TEACHER_NAME, STUDENT_NAME, TEACHER_EPOCHS, STUDENT_EPOCHS,  # noqa: E402
                        DISTILL_T, DISTILL_ALPHA, W2V_DIM, FFN_HIDDEN, FFN_EPOCHS, MAX_LEN)
from src.results_io import load_results, MODEL_ORDER  # noqa: E402

INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
LIGHT = RGBColor(0xF1, 0xF0, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SECTION_COLOR = {"Introduction": BLUE, "Development": RGBColor(0x1B, 0xAF, 0x7A), "Conclusion": ORANGE}

info = json.load(open(ROOT / "slides_info.json", encoding="utf-8"))
R = load_results()
M = R.get("models", {})
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def pct(v):
    return "TBD" if v is None else f"{100 * v:.1f}%"


def g(mid, key, default=None):
    return M.get(mid, {}).get(key, default)


def tb(slide, x, y, w, h, text, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.color.rgb, r.font.bold = Pt(size), color, bold
        p.space_after = Pt(6)
    return box


def bullets(slide, x, y, w, h, items, size=16):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        sub = it.startswith("  ")
        r = p.add_run()
        r.text = ("– " if sub else "• ") + it.strip()
        r.font.size = Pt(size - 2 if sub else size)
        r.font.color.rgb = MUTED if sub else INK
        p.level = 1 if sub else 0
        p.space_after = Pt(5)
    return box


def new_slide(title, section):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = SECTION_COLOR[section]; bar.line.fill.background()
    tb(s, 0.5, 0.3, 9.5, 0.4, section.upper(), 12, SECTION_COLOR[section], True)
    tb(s, 0.5, 0.6, 12.3, 0.8, title, 28, INK, True)
    n = len(prs.slides)
    tb(s, 11.8, 7.0, 1.2, 0.3, f"{n}", 11, MUTED, align=PP_ALIGN.RIGHT)
    tb(s, 0.5, 7.0, 9, 0.3, "SWE015 · Customer Support Ticket Router", 11, MUTED)
    return s


def table(slide, x, y, w, rows, col_w, size=12, highlight_row=None, row_h=0.36):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                               Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(size)
            p.runs[0].font.bold = i == 0
            p.runs[0].font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = INK if i == 0 else (RGBColor(0xFD, 0xE7, 0xDC) if i == highlight_row else
                                                         (LIGHT if i % 2 == 0 else WHITE))
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
    return t


# ---------------------------------------------------------------- 1. title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = INK; bg.line.fill.background()
tb(s, 0.8, 1.2, 11.5, 0.5, info["course"], 18, RGBColor(0xC3, 0xC2, 0xB7))
tb(s, 0.8, 1.8, 11.5, 1.2, "Customer Support Ticket Router", 44, WHITE, True)
tb(s, 0.8, 2.75, 11.5, 0.6, info["use_case"] + " · classical → neural → Transformer → quantized Transformer",
   18, RGBColor(0xF0, 0xEF, 0xEC))
tb(s, 0.8, 4.0, 11.5, 2.5, [f"Name: {info['student_name']}", f"Student ID: {info['student_id']}",
                             f"Repository: {info['repo_url']}", f"Demo video: {info['video_url']}",
                             f"Model weights: {info['hf_url']}"], 18, WHITE)

# ---------------------------------------------------------------- 2. intro
s = new_slide("Problem, users and what success looks like", "Introduction")
bullets(s, 0.5, 1.5, 6.3, 5.3, [
    "A home-appliance brand gets thousands of support messages a day in English and Turkish",
    "  e.g. “bulaşık makinem su boşaltmıyor, model DW-6534”",
    "Manual triage is slow → customers wait, messages land at the wrong team",
    "Users: the support centre (routing) and the customer (instant answer for small talk)",
    "Goal: route every message instantly to the right team; send unsure / out-of-scope messages to a human agent",
], 17)
bullets(s, 7.0, 1.5, 5.8, 5.3, [
    "Success criteria (measured on one held-out test set):",
    "  high in-scope intent accuracy (156 intents)",
    "  high out-of-scope recall (unknown requests → human, never to a wrong team)",
    "  real-time on CPU: tens of milliseconds per message",
    "  small model that is cheap to deploy",
    "Approach: 4-model ladder – classical baseline, own neural network, fine-tuned BERT, distilled + INT8-quantized BERT (used by the app)",
], 17)

# ---------------------------------------------------------------- 3. data
d = R.get("data", {})
s = new_slide("Data: sources, size, splits, preprocessing", "Development")
rows = [["Split", "Total", "In-scope", "Out-of-scope", "of which appliance (ours)"]]
for sp in ["train", "val", "test"]:
    x = d.get(sp, {})
    rows.append([sp, x.get("total", "TBD"), x.get("in_scope", "TBD"), x.get("oos", "TBD"), x.get("appliance", "TBD")])
table(s, 0.5, 1.6, 6.4, rows, [1.0, 1.0, 1.2, 1.5, 1.7], 13)
bullets(s, 0.5, 3.3, 6.4, 3.6, [
    "CLINC150 “full” (Larson et al., EMNLP 2019, CC BY 3.0): 150 intents in 10 domains + out-of-scope class",
    f"+ 100 hand-written appliance messages (EN + TR), 5 intents, 12/3/5 per intent → {d.get('num_intents', 156)} intents total",
    "Unlabelled CLINC Wikipedia sentences (14.7k) used only to train our word2vec",
    "CLINC domain → support team; oos / low confidence → human agent",
], 14)
bullets(s, 7.2, 1.6, 5.6, 5.3, [
    "Same preprocessing for every model (src/preprocess.py):",
    "  Turkish-aware lower-casing + ASCII folding (ş→s, ı→i …)",
    "  regex: URLs, e-mails, repeated letters (“sooo”→“soo”), punctuation",
    "  regex extraction of model codes (DW-6534 → token “modelcode”)",
    "  minimum-edit-distance correction of out-of-vocabulary words against a product lexicon (“dishwaser”→“dishwasher”)",
    "Thresholds τ are tuned on val only; test is touched once per model",
], 14)

# ---------------------------------------------------------------- 4. architecture
s = new_slide("Architecture (matches app.py / src/router.py)", "Development")
arch = DOCS_DIR / "architecture.png"
if arch.exists():
    s.shapes.add_picture(str(arch), Inches(1.2), Inches(1.35), height=Inches(5.6))

# ---------------------------------------------------------------- 5. technique map
s = new_slide("Technique map – which component uses which technique", "Development")
table(s, 0.5, 1.5, 12.3, [
    ["Area", "Technique", "Component in the application", "Code"],
    ["A · Neural networks", "Feedforward net, SGD + momentum, hand-written backprop, dropout, He init",
     "Neural-network rung: averaged word2vec → 512 ReLU → softmax", "src/ffn_np.py"],
    ["B · Classical NLP", "Regex normalization + model-code extraction", "Preprocessing of every message", "src/preprocess.py"],
    ["B · Classical NLP", "Minimum edit distance (Levenshtein) spelling correction", "Misspelled product names", "src/preprocess.py"],
    ["B · Classical NLP", "ELIZA-style regex rules + reflection", "Small-talk replies", "src/eliza.py"],
    ["B · Classical NLP", "tf-idf (word 1-2 + char 2-4 grams) + logistic regression", "Classical baseline rung", "src/baseline.py"],
    ["B · Classical NLP", "word2vec skip-gram with negative sampling (own NumPy)", "Input features of the FFN", "src/word2vec_np.py"],
    ["C · Transformers", "BERT-base fine-tuned with a classification head; WordPiece tokenizer",
     "Transformer rung (teacher)", "src/transformer.py"],
    ["D · Efficient models", "Knowledge distillation (soft targets, T=2) into 4-layer BERT-small",
     "Student model", "src/transformer.py"],
    ["D · Efficient models", "Post-training dynamic INT8 quantization + size/latency benchmark",
     "Quantized Transformer → used by the app", "src/transformer.py, src/benchmark.py"],
], [2.0, 4.3, 3.9, 2.1], 12, highlight_row=9, row_h=0.5)

# ---------------------------------------------------------------- 6. models
s = new_slide("The four models and how each was trained", "Development")
table(s, 0.5, 1.5, 12.3, [
    ["Rung", "Model", "Training"],
    ["1 · Classical baseline", "TF-IDF + logistic regression", "sklearn, C=10, L2; word 1-2-grams + char 2-4-grams; ~1 min on CPU"],
    ["2 · Neural network", f"Own word2vec ({W2V_DIM}d) + FFN ({FFN_HIDDEN} hidden)",
     f"word2vec: SGNS, window 5, 5 negatives, 15 epochs (NumPy). FFN: {FFN_EPOCHS} epochs mini-batch SGD, momentum 0.9, "
     "cosine LR, dropout 0.3, L2 1e-4"],
    ["3 · Transformer", TEACHER_NAME.split("/")[-1] + f" (≈{g('bert_base', 'params_m', 110)}M params)",
     f"Fine-tuned {TEACHER_EPOCHS} epochs, AdamW lr 4e-5, warm-up 10 %, batch 32, max {MAX_LEN} tokens, fp16 on GPU, best epoch on val"],
    ["4 · Quantized Transformer", STUDENT_NAME.split("/")[-1] + f" (≈{g('student_fp32', 'params_m', 29)}M) → INT8",
     f"Distilled from #3 for {STUDENT_EPOCHS} epochs: L = {DISTILL_ALPHA}·CE + {1 - DISTILL_ALPHA}·T²·KL, T={DISTILL_T}; "
     "then dynamic INT8 quantization of all Linear layers"],
], [2.4, 3.6, 6.3], 13, highlight_row=4, row_h=0.75)
tb(s, 0.5, 5.6, 12.3, 1.2, ["All models output a probability for 156 intents (incl. “oos”). If max-probability < τ "
                            "(τ tuned per model on the validation split) the message is treated as out-of-scope → human agent."],
   14, MUTED)

# ---------------------------------------------------------------- 7. quantization
s = new_slide("Distillation + quantization method", "Development")
bullets(s, 0.5, 1.5, 6.5, 5.3, [
    "Knowledge distillation (Hinton et al., 2015)",
    "  teacher logits on the training set are soft targets (T = 2)",
    "  student learns from soft targets + true labels",
    "Post-training dynamic quantization (PyTorch)",
    "  nn.Linear weights stored as INT8 (per-tensor scale + zero-point)",
    "  activations quantized on the fly at inference; embeddings and LayerNorm stay FP32",
    "  no retraining and no calibration data needed",
    "The app reloads the saved INT8 state-dict (verified: reloaded model is the one evaluated)",
], 15)
rows = [["Model", "Precision", "Overall acc.", "Size (MB)", "Latency (ms)"]]
for mid in ["bert_base", "bert_base_int8", "student_fp32", "student_int8"]:
    rows.append([g(mid, "name", mid), g(mid, "precision", "-"), pct(g(mid, "overall_acc")),
                 g(mid, "size_mb", "TBD"), "TBD" if g(mid, "latency_ms") is None else f"{g(mid, 'latency_ms'):.1f}"])
table(s, 7.2, 1.6, 5.6, rows, [2.0, 1.0, 0.9, 0.8, 0.9], 11, highlight_row=4, row_h=0.5)
if g("bert_base", "size_mb") and g("student_int8", "size_mb"):
    tb(s, 7.2, 4.5, 5.6, 2, [f"Teacher → distilled INT8: {g('bert_base', 'size_mb') / g('student_int8', 'size_mb'):.1f}× smaller, "
                             f"{g('bert_base', 'latency_ms') / g('student_int8', 'latency_ms'):.1f}× faster, "
                             f"{100 * (g('bert_base', 'overall_acc') - g('student_int8', 'overall_acc')):+.1f} accuracy points lost "
                             "(negative = gained)."], 14, INK, True)

# ---------------------------------------------------------------- 8. results table
s = new_slide("Results on the same held-out test set (R4)", "Development")
rows = [["Model", "Role", "Precision", "In-scope acc.", "OOS recall", "Overall acc.", "Size (MB)", "Latency (ms)"]]
hl = None
for mid in MODEL_ORDER:
    m = M.get(mid)
    if m:
        rows.append([m["name"], m["role"], m["precision"], pct(m["in_scope_acc"]), pct(m["oos_recall"]),
                     pct(m["overall_acc"]), f"{m['size_mb']:.1f}", f"{m['latency_ms']:.2f}"])
        if mid == "student_int8":
            hl = len(rows) - 1
    else:
        rows.append([mid, "not run yet", "-", "TBD", "TBD", "TBD", "TBD", "TBD"])
table(s, 0.5, 1.5, 12.3, rows, [2.6, 2.4, 1.3, 1.2, 1.1, 1.2, 1.1, 1.4], 12, highlight_row=hl, row_h=0.48)
runs = next((m["latency_runs"] for m in M.values() if "latency_runs" in m), 200)
tb(s, 0.5, 5.2, 12.3, 1.6, [f"Test set: {d.get('test', {}).get('in_scope', '?')} in-scope + {d.get('test', {}).get('oos', '?')} "
                            "out-of-scope messages. Latency = mean wall-clock time for ONE message incl. preprocessing, "
                            f"averaged over {runs} runs after 10 warm-up runs.",
                            f"Hardware: {R.get('hardware', 'TBD')}.",
                            "Size = file size on disk (tf-idf: vectorizers + LR; FFN: word2vec + weights; Transformers: weights)."],
   12, MUTED)

# ---------------------------------------------------------------- 9. chart
s = new_slide("Accuracy vs. size vs. speed", "Development")
chart = RESULTS_DIR / "results_chart.png"
if chart.exists():
    s.shapes.add_picture(str(chart), Inches(0.5), Inches(1.6), width=Inches(12.3))


# ---------------------------------------------------------------- 10. conclusion
def best(key, reverse=True):
    have = [m for m in MODEL_ORDER if m in M]
    if not have:
        return None
    return sorted(have, key=lambda k: M[k][key], reverse=reverse)[0]


s = new_slide("What the comparison showed", "Conclusion")
items = []
if M:
    b_acc, b_oos, fast, small = best("overall_acc"), best("oos_recall"), best("latency_ms", False), best("size_mb", False)
    items.append(f"Highest overall accuracy: {M[b_acc]['name']} ({pct(M[b_acc]['overall_acc'])}); "
                 f"highest OOS recall: {M[b_oos]['name']} ({pct(M[b_oos]['oos_recall'])})")
    items.append(f"Fastest: {M[fast]['name']} ({M[fast]['latency_ms']:.2f} ms); smallest: {M[small]['name']} ({M[small]['size_mb']:.1f} MB)")
if "tfidf_lr" in M and "w2v_ffn" in M:
    diff = 100 * (M["tfidf_lr"]["in_scope_acc"] - M["w2v_ffn"]["in_scope_acc"])
    items.append(f"Baseline vs. own FFN: tf-idf is {diff:+.1f} points in-scope accuracy vs. the FFN – averaging word vectors "
                 "throws away word order and the embeddings are trained on a small corpus; the FFN wins on size and speed")
if "bert_base" in M and "tfidf_lr" in M:
    diff = 100 * (M["bert_base"]["overall_acc"] - M["tfidf_lr"]["overall_acc"])
    verdict = ("contextual embeddings + pre-training on large text pay off" if diff > 0 else
               "the Transformer did NOT beat the baseline – a valid finding: short, keyword-heavy messages favour n-grams")
    items.append(f"BERT-base vs. baseline: {diff:+.1f} points overall accuracy – {verdict}; "
                 f"cost: {M['bert_base']['latency_ms'] / max(M['tfidf_lr']['latency_ms'], 1e-9):.0f}× slower, "
                 f"{M['bert_base']['size_mb']:.0f} MB")
if "student_int8" in M and "bert_base" in M:
    items.append(f"Distillation + INT8 keeps {pct(M['student_int8']['overall_acc'])} overall accuracy "
                 f"(teacher {pct(M['bert_base']['overall_acc'])}) at {M['student_int8']['latency_ms']:.1f} ms per message → "
                 "the best accuracy/speed trade-off of the Transformers, so the app uses it")
if "bert_base_int8" in M and "bert_base" in M:
    items.append(f"INT8 alone on BERT-base: {M['bert_base']['size_mb']:.0f} → {M['bert_base_int8']['size_mb']:.0f} MB, "
                 f"accuracy {pct(M['bert_base']['overall_acc'])} → {pct(M['bert_base_int8']['overall_acc'])} "
                 "(embeddings stay FP32, so the size gain is < 4×)")
if not items:
    items = ["Run `python train_all.py` – this slide is filled automatically from results/results.json"]
bullets(s, 0.5, 1.5, 12.3, 5.3, items, 16)

# ---------------------------------------------------------------- 11. limitations
s = new_slide("Limitations, a failure example and next steps", "Conclusion")
fail_src = next((m for m in ["student_int8", "bert_base", "tfidf_lr"] if m in M), None)
fails = M[fail_src]["errors"] if fail_src else []
pick = []
for cond in [lambda e: e["true"].startswith("appliance") or e["pred"].startswith("appliance"),
             lambda e: e["true"] == "oos", lambda e: e["true"] != "oos" and e["pred"] == "oos",
             lambda e: True]:
    e = next((e for e in fails if cond(e) and e not in pick), None)
    if e and len(pick) < 3:
        pick.append(e)
rows = [["Message (test set)", "True intent", f"Predicted ({M[fail_src]['name'] if fail_src else '-'})"]]
rows += [[e["text"][:70], e["true"], e["pred"]] for e in pick] or [["TBD", "-", "-"]]
table(s, 0.5, 1.5, 12.3, rows, [7.1, 2.5, 2.7], 12, row_h=0.42)
bullets(s, 0.5, 3.5, 6.0, 3.4, [
    "Limitations",
    "  CLINC150 domains are only a proxy for real appliance teams",
    "  only 100 Turkish/appliance examples; English BERT handles Turkish poorly",
    "  one confidence threshold; near-domain unknown requests can still be routed with high confidence",
    "  latency measured on one CPU type",
], 14)
bullets(s, 6.8, 3.5, 6.0, 3.4, [
    "Next steps",
    "  collect real tickets; multilingual model (XLM-R / BERTurk) or MASSIVE-TR data",
    "  ONNX Runtime export + static quantization of embeddings",
    "  calibrate confidences (temperature scaling) for better OOS detection",
    "  active learning from human-agent corrections",
], 14)

# ---------------------------------------------------------------- 12. AI usage
s = new_slide("AI usage statement (R8)", "Conclusion")
table(s, 0.5, 1.5, 12.3, [["Tool / who", "Used for"]] + [list(r) for r in info["ai_tools"]], [2.6, 9.7], 14, row_h=1.0)
tb(s, 0.5, 4.6, 12.3, 2, ["All numbers in this presentation were produced by my own run of train_all.py and are read "
                          "automatically from results/results.json – none were copied from an AI answer, paper or model card.",
                          "Data sources: CLINC150 (Larson et al., 2019, CC BY 3.0); pretrained checkpoints "
                          f"{TEACHER_NAME} and {STUDENT_NAME} (Hugging Face Hub, Apache-2.0)."], 14, MUTED)

# ---------------------------------------------------------------- save
DOCS_DIR.mkdir(exist_ok=True)
out = DOCS_DIR / "presentation.pptx"
prs.save(out)
print("saved", out, f"({len(prs.slides)} slides)")
soffice = shutil.which("soffice") or shutil.which("libreoffice")
if soffice:
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(DOCS_DIR), str(out)],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("saved", DOCS_DIR / "presentation.pdf")
else:
    print("LibreOffice not found - open the .pptx in PowerPoint / Google Slides and export as PDF.")
